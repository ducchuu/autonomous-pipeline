"""DroneTaskEnv: a DirectRLEnv that flies this project's real 5-inch
quadcopter (DRONE_SPEC.md) inside a simulated 8m x 5m x 2.5m OptiTrack
cage, through one of three trainable tasks (hover / figure-8 / shuttle-run),
all sharing the same ground-idle -> vertical-only-ascent takeoff phase.

CONTROL PIPELINE (confirmed with user 2026-08-18, redesigned 2026-08-19 to
full ACRO/rate mode on all 3 axes -- see attitude_controller.py module
docstring for the full rationale): the policy does NOT output per-motor
thrust. It outputs 4 values in [-1, 1] representing RC-style [roll, pitch,
yaw, throttle] channels -- exactly what the real ground PC will send over
ExpressLRS to the flight controller. Inside the sim, those channels are run
through a simulated FC inner loop (attitude_controller.compute_desired_wrench,
a stateful rate-mode P-I-D-FF loop wired via self._rate_controller_state) to
get a desired body torque + collective thrust, which is then inverted into
per-motor thrusts (physics.inverse_mixer) before being applied as a body
force/torque, exactly mirroring how the real FC's own PID loop would turn
the same RC channels into motor commands.

OBSERVATION VECTOR v2 (18-dim, NO battery signals -- confirmed with user
2026-08-18: at inference time the ONLY sensor feeding the policy is
OptiTrack Motive pose, differentiated for velocity; the FC's onboard IMU is
used only for the FC's own internal stabilization and is never seen by the
policy, so training must not let the policy depend on anything OptiTrack
can't provide):
    [0:3]   position error to the CURRENT active target (task target during
            the task phase, ascent waypoint during takeoff), world frame (m)
    [3:7]   body orientation quaternion (w, x, y, z)
    [7:10]  linear velocity, world frame (m/s)
    [10:13] angular velocity, body frame (rad/s)
    [13:17] previous action (the 4 RC-style channels actually sent last
            step) -- helps the policy account for the one-step actuation
            delay inherent in any real control loop
    [17]    phase flag: 0.0 during takeoff, 1.0 during the task phase --
            lets a single policy know which behavior is expected right now
            (a real inference-time script would set this identically)

Battery physics (battery.py) is still simulated and still drives crash
termination (a real flight would also lose control on a dead battery), but
it is intentionally NOT exposed in the observation -- it's a
privileged/training-time-only signal.

Structure follows the official Direct RL env tutorial's split between
``_pre_physics_step`` (process the raw policy action once per control step)
and ``_apply_action`` (apply forces every physics sub-step, called
``decimation`` times per control step):
https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html

This file is intentionally thin -- physics math lives in physics.py,
attitude-control math lives in attitude_controller.py, task targets live in
task_targets.py, the takeoff state machine lives in flight_phase.py, battery
math lives in battery.py, and reward math lives in rewards.py. This file's
job is only to wire those pieces to the Isaac Lab scene/asset API.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from . import attitude_controller, domain_randomization as dr, physics, rewards
from .battery import BatteryModel
from .drone_navigation_env_cfg import DroneTaskEnvCfg
from .flight_phase import PHASE_TASK, FlightPhaseState, active_target, takeoff_target
from .flight_phase import update as update_flight_phase
from .task_targets import get_task_target_fn

OBS_DIM = 18


class DroneTaskEnv(DirectRLEnv):
    """Shared env implementation for all three tasks. Which task actually
    runs is entirely determined by `cfg.task_mode` (set by the DroneHoverEnvCfg
    / DroneFigure8EnvCfg / DroneShuttleRunEnvCfg subclass used at gym.make
    time) -- there is exactly one env class, not one per task, so the
    shared takeoff phase and control pipeline can never drift between tasks.
    """

    cfg: DroneTaskEnvCfg

    def __init__(self, cfg: DroneTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._battery = BatteryModel(num_envs=self.num_envs, device=self.device)
        self._flight_phase = FlightPhaseState.create(num_envs=self.num_envs, device=self.device)
        self._task_target_fn = get_task_target_fn(self.cfg.task_mode)

        self._last_thrusts_n = torch.zeros((self.num_envs, 4), device=self.device)
        self._last_electrical_power_w = torch.zeros(self.num_envs, device=self.device)
        self._last_terminal_voltage_v = torch.zeros(self.num_envs, device=self.device)
        # self._actions always holds the action most recently applied to
        # the sim -- it doubles as the "previous action" observation slot
        # for whichever action the policy is about to choose next.
        self._actions = torch.zeros((self.num_envs, 4), device=self.device)

        self._control_dt_s = self.cfg.sim.dt * self.cfg.decimation

        # --- domain randomization state (see domain_randomization.py) ---
        # cfg.domain_randomization holds the per-component enable flags;
        # this env still SAMPLES/tracks every quantity below regardless of
        # the flags (cheap), and only gates whether each effect is actually
        # APPLIED to the physics/observations -- so flipping a flag never
        # requires touching this state-management code.
        self._dr_cfg = self.cfg.domain_randomization
        dtype = torch.float32
        self._mass_scale = torch.ones(self.num_envs, device=self.device, dtype=dtype)
        self._com_offset_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=dtype)
        self._motor_efficiency = torch.ones((self.num_envs, 4), device=self.device, dtype=dtype)
        self._action_delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        # Action history ring buffer for latency emulation. Neutral action
        # (roll=pitch=yaw=0, throttle=-1 idle) fills unused history slots so
        # a freshly-reset env doesn't get a spurious full-throttle command
        # before it has taken any real action.
        self._action_history = self._neutral_action_history()
        self._wind_gust_state = dr.WindGustState.create(self.num_envs, self.device, dtype)

        # --- rate-mode (ACRO, all 3 axes) inner-loop controller state --
        # NOT domain randomization -- this is the simulated flight
        # controller's own persistent PID state (integral accumulator,
        # previous rate/target for D-on-gyro and feedforward). See
        # attitude_controller.py "ARCHITECTURE HISTORY" (2026-08-19 ACRO
        # redesign) for why this replaced the old stateless angle-mode PD
        # controller. Reset per-env in _reset_idx, NOT in
        # _resample_domain_randomization.
        self._rate_controller_state = attitude_controller.RateControllerState.create(
            self.num_envs, self.device, dtype
        )
        # Resample every quantity for ALL envs once at startup (mirrors the
        # per-env resampling that _reset_idx does on every subsequent reset).
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_domain_randomization(all_env_ids)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        self.robot = self.scene["robot"]

    # ------------------------------------------------------------------
    # Domain randomization helpers
    # ------------------------------------------------------------------

    def _neutral_action_history(self) -> torch.Tensor:
        """A full ACTION_HISTORY_LEN-deep history buffer pre-filled with the
        neutral RC action (level attitude, idle throttle) for every env.
        """
        neutral = torch.tensor([0.0, 0.0, 0.0, -1.0], device=self.device)
        return neutral.expand(self.num_envs, dr.ACTION_HISTORY_LEN, 4).clone()

    def _resample_domain_randomization(self, env_ids: torch.Tensor) -> None:
        """(Re)sample every per-episode domain-randomization quantity for
        the given envs. Called once for all envs at __init__ and again for
        the reset subset on every _reset_idx, regardless of which
        components are currently enabled in cfg.domain_randomization (cheap,
        and keeps this bookkeeping in one place instead of scattered behind
        per-flag branches).
        """
        n = env_ids.shape[0]
        dtype = self._mass_scale.dtype
        self._mass_scale[env_ids] = dr.sample_mass_scale(n, self.device, dtype)
        self._com_offset_xy[env_ids] = dr.sample_com_offset_xy(n, self.device, dtype)
        self._motor_efficiency[env_ids] = dr.sample_motor_efficiency(n, self.device, dtype)
        self._action_delay_steps[env_ids] = dr.sample_action_delay_steps(n, self.device)

        neutral = torch.tensor([0.0, 0.0, 0.0, -1.0], device=self.device, dtype=dtype)
        self._action_history[env_ids] = neutral.expand(n, dr.ACTION_HISTORY_LEN, 4).clone()
        self._wind_gust_state.reset(env_ids)

    # ------------------------------------------------------------------
    # Action processing (once per control step)
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called once per control step (every `decimation` physics steps).

        Converts the raw [-1, 1] RC-style action [roll, pitch, yaw,
        throttle] through the simulated flight-controller inner loop
        (attitude_controller) into a desired torque + collective thrust,
        then inverts that wrench into per-motor thrusts (physics.
        inverse_mixer) and the resulting body-frame force/torque, cached
        for `_apply_action` to apply on every physics sub-step in between.
        """
        # self._actions holds the RAW action the policy just commanded --
        # this is what goes into the "previous action" observation slot,
        # since a real policy remembers what IT commanded, not whatever
        # delayed/latent action the flight controller ends up executing.
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # --- action latency: push the raw action into the per-env ring
        # buffer and read back whichever one THIS env's sampled radio-link
        # delay says should actually reach the flight controller this step.
        if self._dr_cfg.enable_action_latency:
            self._action_history = dr.push_action_history(self._action_history, self._actions)
            applied_action = dr.read_delayed_action(self._action_history, self._action_delay_steps)
        else:
            applied_action = self._actions

        root_state = self.robot.data.root_state_w
        current_ang_vel_b = root_state[:, 10:13]

        # Full ACRO/rate mode on all 3 axes (2026-08-19) -- like a real
        # ACRO-mode FC, the inner loop only ever reads the gyro, never an
        # absolute attitude estimate, so no quaternion is passed here. See
        # attitude_controller.py "ARCHITECTURE HISTORY". The controller is
        # now stateful (integral/D/FF need persistent state across control
        # steps), hence `self._rate_controller_state` + `self._control_dt_s`.
        desired_force_z, desired_torque = attitude_controller.compute_desired_wrench(
            applied_action, current_ang_vel_b, self._rate_controller_state, self._control_dt_s
        )
        self._last_thrusts_n = physics.inverse_mixer(desired_force_z, desired_torque)

        # --- per-motor efficiency: the mixer's commanded thrust is what the
        # FC ASKS each motor for; real motors deliver slightly more/less.
        if self._dr_cfg.enable_motor_efficiency:
            delivered_thrusts_n = dr.apply_motor_efficiency(self._last_thrusts_n, self._motor_efficiency)
        else:
            delivered_thrusts_n = self._last_thrusts_n

        forces_b, torques_b = physics.mixer_forces_and_torques(delivered_thrusts_n)

        # --- mass randomization (emulated via force/torque scaling -- see
        # domain_randomization.py module docstring for why this is exactly
        # equivalent to a real body of mass MASS_TOTAL_KG * mass_scale).
        if self._dr_cfg.enable_mass_randomization:
            forces_b, torques_b = dr.apply_mass_scale_to_wrench(forces_b, torques_b, self._mass_scale)

        # --- center-of-mass offset: extra disturbance torque from applying
        # collective thrust off-center, computed from the (un-mass-scaled)
        # commanded thrust so the disturbance magnitude tracks the actual
        # applied force.
        if self._dr_cfg.enable_com_offset:
            torques_b = torques_b + dr.com_offset_disturbance_torque(self._com_offset_xy, forces_b)

        # --- wind gusts: transient external push/twist, independent of
        # anything the policy commanded.
        if self._dr_cfg.enable_wind_gusts:
            gust_force_b, gust_torque_b = dr.update_wind_gusts(self._wind_gust_state, self._control_dt_s)
            forces_b = forces_b + gust_force_b
            torques_b = torques_b + gust_torque_b

        self._forces_b, self._torques_b = forces_b, torques_b
        # Electrical power draw reflects what the FC actually asked the
        # motors for (pre-efficiency-scaling commanded thrust), matching a
        # real ESC's current-draw telemetry which tracks commanded duty
        # cycle, not the resulting (possibly efficiency-degraded) thrust.
        self._last_electrical_power_w = physics.induced_electrical_power(self._last_thrusts_n)

    def _apply_action(self) -> None:
        """Called every physics sub-step. Applies the cached body-frame
        force/torque to the drone rigid body.
        """
        self.robot.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self._forces_b,
            torque_data=self._torques_b,
            position_data=None,
            indices=self._robot_indices(),
            is_global=False,
        )



    def _robot_indices(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Flight-phase bookkeeping (shared takeoff -> task state machine)
    # ------------------------------------------------------------------

    def _current_target_and_phase(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the takeoff/task phase state machine by one control step
        and return (current active target, is_task_phase bool mask).

        Call exactly ONCE per control step (from `_get_rewards`, which Isaac
        Lab's DirectRLEnv guarantees runs after `_apply_action` and before
        `_get_observations` reads phase-derived values this same step) --
        NOT from `_get_observations` too, to avoid double-advancing the
        state machine if both are called in the same step.
        """
        root_state = self.robot.data.root_state_w
        pos_w = root_state[:, 0:3]
        quat_w = root_state[:, 3:7]
        lin_vel_w = root_state[:, 7:10]

        tilt_rad = self._tilt_from_quat(quat_w)
        update_flight_phase(self._flight_phase, pos_w, lin_vel_w, tilt_rad, self._control_dt_s)

        target = active_target(self._flight_phase, self._task_target_fn)
        is_task_phase = self._flight_phase.phase == PHASE_TASK
        return target, is_task_phase

    @staticmethod
    def _tilt_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
        """Angle (radians) between the body +Z axis and world +Z, i.e. how
        far the drone is tilted from level, regardless of yaw.
        """
        w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
        up_z = 1.0 - 2.0 * (x**2 + y**2)
        return torch.acos(up_z.clamp(-1.0, 1.0))

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        root_state = self.robot.data.root_state_w
        pos_w = root_state[:, 0:3]
        quat_w = root_state[:, 3:7]
        lin_vel_w = root_state[:, 7:10]
        ang_vel_b = root_state[:, 10:13]

        # --- observation noise: emulates real OptiTrack pose jitter (and
        # the amplified noise a real velocity-by-differentiation pipeline
        # would see) -- applied ONLY here, never to the true state used for
        # rewards/termination/phase logic, exactly like a real policy that
        # only ever sees noisy sensor data.
        if self._dr_cfg.enable_observation_noise:
            pos_w = dr.add_gaussian_noise(pos_w, dr.OBS_POS_NOISE_STD_M)
            quat_w = dr.perturb_quat(quat_w, dr.OBS_ROT_NOISE_STD_RAD)
            lin_vel_w = dr.add_gaussian_noise(lin_vel_w, dr.OBS_LIN_VEL_NOISE_STD_MPS)
            ang_vel_b = dr.add_gaussian_noise(ang_vel_b, dr.OBS_ANG_VEL_NOISE_STD_RAD_S)

        # NOTE: the phase state machine is advanced in _get_rewards (called
        # first each step by DirectRLEnv); this just reads the already
        # up-to-date target/phase for this step's observation.
        target_w = active_target(self._flight_phase, self._task_target_fn)
        is_task_phase = (self._flight_phase.phase == PHASE_TASK).float().unsqueeze(-1)

        rel_pos = target_w - pos_w

        obs = torch.cat(
            [rel_pos, quat_w, lin_vel_w, ang_vel_b, self._actions, is_task_phase], dim=-1
        )
        assert obs.shape[-1] == OBS_DIM, f"observation dim mismatch: {obs.shape[-1]} != {OBS_DIM}"
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        target_w, is_task_phase = self._current_target_and_phase()

        root_state = self.robot.data.root_state_w
        pos_w = root_state[:, 0:3]
        quat_w = root_state[:, 3:7]
        lin_vel_w = root_state[:, 7:10]
        ang_vel_b = root_state[:, 10:13]

        pos_error_norm = torch.linalg.norm(target_w - pos_w, dim=-1)
        lin_vel_norm = torch.linalg.norm(lin_vel_w, dim=-1)
        ang_vel_norm_sq = torch.sum(ang_vel_b**2, dim=-1)

        # Takeoff-only stability terms: lateral drift from the (x=0,y=0)
        # ascent axis and tilt, computed unconditionally (cheap) and masked
        # by the phase flag inside compute_total_reward.
        ascent_target = takeoff_target(self.num_envs, self.device, pos_w.dtype)
        lateral_pos_error_norm = torch.linalg.norm(
            (ascent_target - pos_w)[:, :2], dim=-1
        )
        tilt_rad = self._tilt_from_quat(quat_w)

        self._last_terminal_voltage_v = self._battery.step(
            self._last_electrical_power_w, dt_s=self._control_dt_s
        )
        crashed = self._is_crashed()

        return rewards.compute_total_reward(
            pos_error_norm=pos_error_norm,
            lin_vel_norm=lin_vel_norm,
            ang_vel_norm_sq=ang_vel_norm_sq,
            electrical_power_w=self._last_electrical_power_w,
            crashed=crashed,
            weight_velocity=self.cfg.reward_weight_velocity,
            weight_angular_rate=self.cfg.reward_weight_angular_rate,
            weight_energy=self.cfg.reward_weight_energy,
            weight_crash=self.cfg.reward_weight_crash,
            is_takeoff_phase=~is_task_phase,
            lateral_pos_error_norm=lateral_pos_error_norm,
            tilt_rad=tilt_rad,
            weight_takeoff_stability=self.cfg.reward_weight_takeoff_stability,
        )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_crashed(self) -> torch.Tensor:
        root_state = self.robot.data.root_state_w
        pos_w = root_state[:, 0:3]
        quat_w = root_state[:, 3:7]

        # OptiTrack cage bounds (re-measured by user 2026-08-19): 8m (X) x
        # 5m (Y) x 2.5m (Z), origin (0,0,0) at floor center -- so X/Y each
        # range +/- world_bounds_x_m / world_bounds_y_m, Z ranges
        # 0..world_bounds_z_m. See drone_navigation_env_cfg.py for the
        # actual bound values and DRONE_SPEC.md section 8 for the history.
        out_of_bounds_xy = (pos_w[:, 0].abs() > self.cfg.world_bounds_x_m) | (
            pos_w[:, 1].abs() > self.cfg.world_bounds_y_m
        )
        out_of_bounds_z = (pos_w[:, 2] < 0.02) | (pos_w[:, 2] > self.cfg.world_bounds_z_m)

        tilt_rad = self._tilt_from_quat(quat_w)
        excessive_tilt = tilt_rad > self.cfg.max_tilt_rad

        battery_dead = self._battery.is_dead(self._last_terminal_voltage_v)

        return out_of_bounds_xy | out_of_bounds_z | excessive_tilt | battery_dead

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        crashed = self._is_crashed()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return crashed, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self._battery.reset(env_ids)
        self._flight_phase.reset(env_ids)

        self._last_thrusts_n[env_ids] = 0.0
        self._last_electrical_power_w[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        # Rate-mode inner-loop controller state (integral accumulator,
        # prev rate/target) must be zeroed on episode reset too -- it is
        # NOT part of domain randomization, so it is reset here directly
        # rather than inside _resample_domain_randomization.
        self._rate_controller_state.reset(env_ids)

        self._resample_domain_randomization(env_ids)
