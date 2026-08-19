"""Domain randomization for sim-to-real transfer.

WHY THIS EXISTS: a policy trained against one perfect, deterministic
physics model will not survive contact with the real drone -- real battery
voltage sag, motor-to-motor thrust mismatch, a center of mass that isn't
exactly where the CAD model says, radio/flight-controller latency, OptiTrack
jitter, and stray air currents (prop wash bouncing off the cage floor/walls)
are all present on every real flight and absent from a naive simulation.
Every randomizer below forces the policy to learn a controller that is
robust to a *range* of plausible real-world conditions instead of
memorizing the exact physics of one simulated drone.

This module is pure torch, isaaclab-free by design (see
tests/test_domain_randomization.py, loaded via the same importlib +
sys.modules pre-registration pattern as physics.py / attitude_controller.py
/ task_targets.py / flight_phase.py), so every randomizer here can be
verified with plain `pytest`, no Isaac Sim required.

WHERE EACH RANDOMIZER PLUGS IN (see drone_navigation_env.py for the exact
wiring):
  - Mass scale        -> _pre_physics_step, scales the applied force/torque
  - CoM offset         -> _pre_physics_step, adds a disturbance torque
  - Motor efficiency   -> _pre_physics_step, scales per-motor thrust
  - Action latency     -> _pre_physics_step, delays which action is applied
  - Observation noise  -> _get_observations only (reward/termination always
                           use the true, noise-free state -- exactly like a
                           real RL loop that only ever sees noisy sensor
                           data, never "ground truth")
  - Wind gusts         -> _pre_physics_step, adds a transient external
                           force/torque

ALL numeric ranges below are reasoned engineering defaults, NOT measured
from your real drone or cage -- tagged [ESTIMATE-VERIFY] throughout, same
convention as physics.py/battery.py/attitude_controller.py. Safe to retune
without touching any other file; see DRONE_SPEC.md section 13.

IMPORTANT MODELING NOTE ON MASS RANDOMIZATION: this project deliberately
does NOT call any Isaac Lab / PhysX API to literally rewrite the simulated
rigid body's mass at reset time, because the exact API for that is one more
version-sensitive surface on an already non-standard Isaac Lab install
("6.1.17", see AGENT_VERIFICATION_LOG.md). Instead, `apply_mass_scale_to_wrench`
uses an equivalent force/torque-scaling trick: dividing the applied
force/torque by `mass_scale` produces exactly the same acceleration response
a real body of mass `MASS_TOTAL_KG * mass_scale` would have under the same
commanded force/torque, while gravity (computed by the physics engine from
the ACTUAL unscaled simulated mass) is left untouched -- so a "heavier"
emulated drone correctly needs more throttle just to hover, and a "lighter"
one correctly needs less. This reproduces the physically-relevant training
signal (the policy must adapt its throttle/attitude response to an unknown
effective mass) without depending on an unverified low-level API. If you
later confirm the real PhysX mass-write API for your install and want a
literal per-env mass, you can add that alongside or instead of this -- see
`.agents/rules/isaac_lab_drone_project.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import physics

# --------------------------------------------------------------------------
# 1. Mass randomization (emulated via force/torque scaling, see module
#    docstring above for the full rationale)
# --------------------------------------------------------------------------

MASS_SCALE_RANGE = (0.90, 1.10)  # [ESTIMATE-VERIFY] +/-10% all-up mass per episode


def sample_mass_scale(num_envs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample one mass-scale factor per env, uniform in MASS_SCALE_RANGE."""
    low, high = MASS_SCALE_RANGE
    return torch.empty(num_envs, device=device, dtype=dtype).uniform_(low, high)


def apply_mass_scale_to_wrench(
    force_b: torch.Tensor, torque_b: torch.Tensor, mass_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale the commanded body-frame force/torque by 1/mass_scale so the
    resulting acceleration matches what a real body of mass
    `MASS_TOTAL_KG * mass_scale` would experience under the same commanded
    force/torque, given the simulated rigid body's actual (unscaled) mass
    and inertia. See module docstring "MODELING NOTE ON MASS RANDOMIZATION".

    force_b:  (num_envs, 3) newtons
    torque_b: (num_envs, 3) N*m
    mass_scale: (num_envs,) unitless, one MASS_SCALE_RANGE sample per env
    """
    inv_scale = (1.0 / mass_scale).unsqueeze(-1)
    return force_b * inv_scale, torque_b * inv_scale


# --------------------------------------------------------------------------
# 2. Center-of-mass offset randomization (physically exact disturbance
#    torque, no PhysX CoM API dependency)
# --------------------------------------------------------------------------

COM_OFFSET_RANGE_M = 0.02  # [ESTIMATE-VERIFY] +/-2cm in body-frame x and y


def sample_com_offset_xy(num_envs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample one (dx, dy) body-frame CoM offset per env, uniform in
    [-COM_OFFSET_RANGE_M, +COM_OFFSET_RANGE_M] on each axis independently.
    """
    return torch.empty((num_envs, 2), device=device, dtype=dtype).uniform_(
        -COM_OFFSET_RANGE_M, COM_OFFSET_RANGE_M
    )


def com_offset_disturbance_torque(
    com_offset_xy: torch.Tensor, force_b: torch.Tensor
) -> torch.Tensor:
    """Extra body-frame torque caused by applying a collective thrust force
    at the drone's TRUE geometric center while its actual mass is
    concentrated `com_offset_xy` away from that point -- i.e. exactly the
    torque = r x F a real drone would feel if its battery/wiring shifted the
    CoM by that offset. Uses the same right-hand-rule sign convention as
    physics.mixer_forces_and_torques (tau_x = +y*Fz, tau_y = -x*Fz) since
    the thrust force here is purely along body +Z, same as that function.

    com_offset_xy: (num_envs, 2) meters, (dx, dy) in the body frame
    force_b:       (num_envs, 3) newtons, body-frame applied force (only
                    the z-component is physically meaningful for this
                    drone's thrust-along-+Z geometry)
    returns: (num_envs, 3) N*m extra (tau_x, tau_y, 0) to add to the
             commanded torque before applying it to the rigid body
    """
    dx = com_offset_xy[..., 0]
    dy = com_offset_xy[..., 1]
    fz = force_b[..., 2]
    tau_x = dy * fz
    tau_y = -dx * fz
    tau_z = torch.zeros_like(fz)
    return torch.stack([tau_x, tau_y, tau_z], dim=-1)


# --------------------------------------------------------------------------
# 3. Per-motor thrust/efficiency randomization
# --------------------------------------------------------------------------

MOTOR_EFFICIENCY_RANGE = (0.85, 1.05)  # [ESTIMATE-VERIFY] per-motor, resampled every episode


def sample_motor_efficiency(num_envs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample 4 independent per-motor efficiency scales per env, uniform in
    MOTOR_EFFICIENCY_RANGE. Independent per motor (not one shared scalar per
    env) so the policy must use attitude feedback to compensate for
    asymmetric motors, not just learn a single global throttle bias.
    """
    low, high = MOTOR_EFFICIENCY_RANGE
    return torch.empty((num_envs, 4), device=device, dtype=dtype).uniform_(low, high)


def apply_motor_efficiency(commanded_thrusts_n: torch.Tensor, efficiency: torch.Tensor) -> torch.Tensor:
    """Scale the mixer's commanded per-motor thrust by each motor's sampled
    efficiency to get the thrust the (imperfect) real motor would actually
    deliver.

    commanded_thrusts_n: (num_envs, 4) newtons, from physics.inverse_mixer
    efficiency:          (num_envs, 4) unitless, one MOTOR_EFFICIENCY_RANGE
                         sample per motor per env
    returns: (num_envs, 4) newtons, the actually-delivered thrust
    """
    return commanded_thrusts_n * efficiency


# --------------------------------------------------------------------------
# 4. Action latency randomization (the #1 sim-to-real killer per the PC ->
#    RadioMaster -> ExpressLRS -> FC -> ESC -> motor pipeline)
# --------------------------------------------------------------------------

ACTION_DELAY_MIN_STEPS = 1
ACTION_DELAY_MAX_STEPS = 3  # [ESTIMATE-VERIFY] at 50Hz control (0.02s/step), 1-3 steps = 20-60ms
ACTION_HISTORY_LEN = ACTION_DELAY_MAX_STEPS + 1


def sample_action_delay_steps(num_envs: int, device: torch.device) -> torch.Tensor:
    """Sample one integer action-delay (in control steps) per env, uniform
    over {ACTION_DELAY_MIN_STEPS, ..., ACTION_DELAY_MAX_STEPS}. Fixed for the
    whole episode (a real radio link's latency is roughly constant within a
    single flight), resampled at every reset.
    """
    return torch.randint(
        ACTION_DELAY_MIN_STEPS,
        ACTION_DELAY_MAX_STEPS + 1,
        (num_envs,),
        device=device,
    )


def push_action_history(history: torch.Tensor, new_action: torch.Tensor) -> torch.Tensor:
    """Roll a new action into the front of a per-env action history buffer,
    dropping the oldest entry.

    history:    (num_envs, ACTION_HISTORY_LEN, 4) -- index 0 = most recent
    new_action: (num_envs, 4)
    returns:    (num_envs, ACTION_HISTORY_LEN, 4) updated history
    """
    return torch.cat([new_action.unsqueeze(1), history[:, :-1, :]], dim=1)


def read_delayed_action(history: torch.Tensor, delay_steps: torch.Tensor) -> torch.Tensor:
    """Read back the action from `delay_steps` control steps ago, per env,
    from a history buffer already updated this step via push_action_history
    (so delay_steps=0 would mean "no delay, use the action just pushed").

    history:     (num_envs, ACTION_HISTORY_LEN, 4)
    delay_steps: (num_envs,) integer, in [0, ACTION_HISTORY_LEN - 1]
    returns:     (num_envs, 4) the delayed action actually applied
    """
    num_envs = history.shape[0]
    env_idx = torch.arange(num_envs, device=history.device)
    return history[env_idx, delay_steps, :]


# --------------------------------------------------------------------------
# 5. Observation noise (OptiTrack realism -- see DRONE_SPEC.md section 11
#    for why battery is excluded from observations; noise applies only to
#    what a real OptiTrack + differentiation pipeline would actually see)
# --------------------------------------------------------------------------

OBS_POS_NOISE_STD_M = 0.002        # [ESTIMATE-VERIFY] OptiTrack marker jitter
OBS_ROT_NOISE_STD_RAD = 0.002      # [ESTIMATE-VERIFY] OptiTrack orientation jitter
OBS_LIN_VEL_NOISE_STD_MPS = 0.05   # [ESTIMATE-VERIFY] amplified by differentiating noisy position
OBS_ANG_VEL_NOISE_STD_RAD_S = 0.05  # [ESTIMATE-VERIFY] amplified by differentiating noisy orientation


def add_gaussian_noise(value: torch.Tensor, std: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise with the given standard deviation,
    elementwise, matching `value`'s shape/device/dtype. std=0.0 is a no-op
    (useful for disabling noise for evaluation/deployment sanity checks).
    """
    if std <= 0.0:
        return value
    return value + torch.randn_like(value) * std


def _quat_multiply_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2, both (num_envs, 4) in (w, x, y, z) order."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def perturb_quat(quat_wxyz: torch.Tensor, noise_std_rad: float) -> torch.Tensor:
    """Compose a small random rotation (small-angle approximation) onto a
    true (w, x, y, z) quaternion to emulate OptiTrack orientation jitter.
    noise_std_rad=0.0 is a no-op (returns quat_wxyz unchanged).

    quat_wxyz: (num_envs, 4)
    returns:   (num_envs, 4), unit-norm
    """
    if noise_std_rad <= 0.0:
        return quat_wxyz
    small_rotvec = torch.randn(
        (*quat_wxyz.shape[:-1], 3), device=quat_wxyz.device, dtype=quat_wxyz.dtype
    ) * noise_std_rad
    half = small_rotvec * 0.5
    noise_quat = torch.cat(
        [torch.ones_like(half[..., :1]), half], dim=-1
    )
    noise_quat = noise_quat / torch.linalg.norm(noise_quat, dim=-1, keepdim=True)
    noisy = _quat_multiply_wxyz(noise_quat, quat_wxyz)
    return noisy / torch.linalg.norm(noisy, dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# 6. External perturbations ("wind gusts" -- turbulent air / prop wash
#    bouncing off the cage floor and walls, especially on a shuttle run)
# --------------------------------------------------------------------------

_HOVER_THRUST_N = physics.MASS_TOTAL_KG * physics.GRAVITY_MPS2

WIND_GUST_MEAN_INTERVAL_S = 3.0        # [ESTIMATE-VERIFY] average time between gusts, per env
WIND_GUST_DURATION_STEPS_RANGE = (5, 15)  # [ESTIMATE-VERIFY] at 0.02s/step: 0.1-0.3s gusts
WIND_GUST_FORCE_MAX_N = 0.15 * _HOVER_THRUST_N   # [ESTIMATE-VERIFY] ~15% of hover thrust, horizontal
WIND_GUST_TORQUE_MAX_NM = 0.05                    # [ESTIMATE-VERIFY] small disturbance torque


@dataclass
class WindGustState:
    """Per-env wind-gust state: how many control steps the current gust
    still has left to run, and the force/torque it applies while active.
    """

    remaining_steps: torch.Tensor  # (num_envs,) int64
    force_b: torch.Tensor  # (num_envs, 3)
    torque_b: torch.Tensor  # (num_envs, 3)

    @classmethod
    def create(cls, num_envs: int, device: torch.device, dtype: torch.dtype) -> "WindGustState":
        return cls(
            remaining_steps=torch.zeros(num_envs, dtype=torch.int64, device=device),
            force_b=torch.zeros((num_envs, 3), device=device, dtype=dtype),
            torque_b=torch.zeros((num_envs, 3), device=device, dtype=dtype),
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self.remaining_steps[env_ids] = 0
        self.force_b[env_ids] = 0.0
        self.torque_b[env_ids] = 0.0


def update_wind_gusts(state: WindGustState, control_dt_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the wind-gust state machine by one control step and return
    the (force_b, torque_b) to apply THIS step (zero where no gust is
    active).

    For envs with no gust currently active (remaining_steps == 0), trigger
    a new gust with probability `control_dt_s / WIND_GUST_MEAN_INTERVAL_S`
    (a standard discrete-time approximation of a Poisson process with the
    stated mean inter-gust interval). For envs with an active gust,
    decrement the remaining duration.
    """
    device = state.remaining_steps.device
    num_envs = state.remaining_steps.shape[0]

    no_gust_active = state.remaining_steps <= 0
    trigger_prob = control_dt_s / WIND_GUST_MEAN_INTERVAL_S
    roll = torch.rand(num_envs, device=device)
    should_trigger = no_gust_active & (roll < trigger_prob)

    if should_trigger.any():
        n_trigger = int(should_trigger.sum().item())
        dtype = state.force_b.dtype

        # Random horizontal push direction, uniform magnitude up to the max.
        angle = torch.rand(n_trigger, device=device, dtype=dtype) * (2.0 * math.pi)
        magnitude = torch.rand(n_trigger, device=device, dtype=dtype) * WIND_GUST_FORCE_MAX_N
        gust_force_x = magnitude * torch.cos(angle)
        gust_force_y = magnitude * torch.sin(angle)
        gust_force_z = torch.zeros(n_trigger, device=device, dtype=dtype)
        new_force = torch.stack([gust_force_x, gust_force_y, gust_force_z], dim=-1)

        new_torque = (
            torch.empty((n_trigger, 3), device=device, dtype=dtype).uniform_(
                -WIND_GUST_TORQUE_MAX_NM, WIND_GUST_TORQUE_MAX_NM
            )
        )

        low, high = WIND_GUST_DURATION_STEPS_RANGE
        new_duration = torch.randint(low, high + 1, (n_trigger,), device=device)

        state.force_b[should_trigger] = new_force
        state.torque_b[should_trigger] = new_torque
        state.remaining_steps[should_trigger] = new_duration

    active = state.remaining_steps > 0
    out_force = torch.where(active.unsqueeze(-1), state.force_b, torch.zeros_like(state.force_b))
    out_torque = torch.where(active.unsqueeze(-1), state.torque_b, torch.zeros_like(state.torque_b))

    state.remaining_steps = torch.clamp(state.remaining_steps - 1, min=0)

    return out_force, out_torque
