"""Configuration for DroneTaskEnv (hover / figure-8 / shuttle-run share one
env class, see __init__.py for the 3 separate gym registrations).

Follows the Isaac Lab DirectRLEnvCfg pattern documented at
https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html
and mirrors the structure of the officially shipped quadcopter task config
(https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env_cfg.py),
adapted to this project's real drone parameters (DRONE_SPEC.md) and reward
function (rewards.py).

IMPORTANT (see .agents/rules/isaac_lab_drone_project.md section on version
verification): the exact import paths below (isaaclab.assets, isaaclab.sim,
isaaclab.envs, isaaclab.scene, isaaclab.utils) match the "main" branch docs
consulted for this project. Your installed Isaac Lab reports an unusual
version string ("6.1.17") that does not match any documented release --
verify these imports resolve in your conda env (`python -c "import
isaaclab.envs"`) before assuming they are correct, and consult
isaaclab.utils.configclass / isaaclab.envs.DirectRLEnvCfg's actual installed
signature if any of this fails to import.
"""

from __future__ import annotations
import os
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass


from . import physics
from .task_targets import (
    FIGURE8_PERIOD_S,
    HOVER_HEIGHT_M,
    SHUTTLE_RUN_SWITCH_PERIOD_S,
    TASK_FIGURE8,
    TASK_HOVER,
    TASK_SHUTTLE_RUN,
)


@configclass
class DomainRandomizationCfg:
    """Per-component on/off switches for domain_randomization.py. All True
    by default for TRAINING (see DroneTaskEnvCfg below). Numeric ranges live
    in domain_randomization.py itself, not here, so there is exactly one
    place to retune them.

    IMPORTANT: use DroneTaskEnvCfg.for_evaluation() (or set every flag below
    to False) when running scripts/sb3/play.py or any other clean rollout
    where you want to see the policy's behavior without added noise/latency
    -- training and "what does this look like" evaluation are different use
    cases and should not share the same randomization settings.
    """

    enable_mass_randomization: bool = True
    enable_com_offset: bool = True
    enable_motor_efficiency: bool = True
    enable_action_latency: bool = True
    enable_observation_noise: bool = True
    enable_wind_gusts: bool = True

    @classmethod
    def all_disabled(cls) -> "DomainRandomizationCfg":
        return cls(
            enable_mass_randomization=False,
            enable_com_offset=False,
            enable_motor_efficiency=False,
            enable_action_latency=False,
            enable_observation_noise=False,
            enable_wind_gusts=False,
        )


MESH_USD_PATH = "{ENV_REGEX_NS}/../../../../../../mesh/5-inch_drone.usdc"
# NOTE: verify this relative path resolves correctly from your actual asset
# root at import time -- Isaac Lab typically resolves relative USD paths
# against the extension's data directory, not the CWD. If the mesh does not
# load, replace this with an absolute path or copy the .usdc into this
# project's own data folder (e.g. source/autonomous_pipeline/data/) and
# reference it from there instead. This is flagged as a TODO for the local
# coding agent to confirm against your real Isaac Lab asset-path conventions.


@configclass
class DroneNavigationSceneCfg(InteractiveSceneCfg):
    """Scene: ground plane, dome light, and one drone rigid body per env."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    # Modeled as a RigidObject (not an Articulation) because the drone body
    # is a single rigid link -- thrust/torque are applied as external
    # wrenches on this one body, there is no internal joint to actuate.
    # See the official quadcopter example for the same modeling choice.
    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../../data/5-inch_drone.usdc",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=physics.MASS_TOTAL_KG,
            ),
        ),
        # Drone starts resting flat on the floor, motors armed/idle, roughly
        # centered in the cage -- matches the real OptiTrack cage setup
        # (re-measured by user 2026-08-19): origin (0,0,0) at floor center,
        # cage is 8m (X) x 5m (Y) x 2.5m (Z). 0.05m clears the props off the
        # ground plane without implying any takeoff progress yet -- ascent to
        # HOVER_HEIGHT_M is entirely the policy/flight_phase's job, not a
        # spawn-time shortcut.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
    )


@configclass
class DroneTaskEnvCfg(DirectRLEnvCfg):
    """Base config shared by all three trainable tasks (hover, figure-8,
    shuttle-run). Each task subclasses this and only overrides `task_mode`
    plus any task-specific tunable below -- world bounds, control pipeline,
    observation layout, takeoff behavior, and reward weights are IDENTICAL
    across tasks by design (see task_targets.py / flight_phase.py module
    docstrings for the shared-takeoff rationale).

    Values here are the ones you tune during training iteration; physical
    constants live in physics.py / battery.py and should stay in sync with
    DRONE_SPEC.md instead of being duplicated here.
    """

    # --- which task this env instance runs (set by the 3 subclasses below) ---
    task_mode: str = TASK_HOVER

    # --- timing (DRONE_SPEC.md section 6) ---
    decimation = 2
    episode_length_s = 15.0
    # Action space: [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd] in [-1, 1],
    # RC-channel style -- NOT direct per-motor thrust. See
    # attitude_controller.py module docstring for the full sim-to-real
    # rationale (confirmed with user 2026-08-18).
    action_space = 4
    # Observation vector v2 (18-dim, NO battery signals -- real inference
    # only has OptiTrack pose+velocity, see drone_navigation_env.py OBS_DIM
    # comment for the exact per-slot layout):
    #   rel_pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + prev_action(4) + phase_flag(1) = 18
    observation_space = 18
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=decimation,
    )

    scene: DroneNavigationSceneCfg = DroneNavigationSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # --- OptiTrack cage / world bounds (re-measured by user 2026-08-19,
    # SUPERSEDES the earlier "10m x 10m x 2.5m" figure used in the
    # 2026-08-18 stretch -- see DRONE_SPEC.md section 8 for the discrepancy
    # note; the earlier figure was never actually "7x7" as the user recalled
    # either, it was documented/derived as 10x10) ---
    # 8m (X, "length") x 5m (Y, "width") x 2.5m (Z, "height"), origin (0,0,0)
    # at floor center. Axis assignment: X=length because the shuttle-run
    # task already runs along X and length is the longer dimension.
    # [PROVISIONAL] exact OptiTrack calibration origin is still an estimate
    # ("roughly cage-centered") -- update these if it turns out not to be
    # floor-centered.
    world_bounds_x_m: float = 4.0   # +/- 4m => 8m total (length)
    world_bounds_y_m: float = 2.5   # +/- 2.5m => 5m total (width)
    world_bounds_z_m: float = 2.5   # 0 (floor) .. 2.5m (ceiling), unchanged

    # --- takeoff phase (shared by all tasks, see flight_phase.py) ---
    hover_height_m: float = HOVER_HEIGHT_M  # 1.0m, confirmed target ascent height

    # --- termination thresholds ---
    max_tilt_rad: float = math.radians(60.0)  # crash if body tilts past this

    # --- reward weights (rewards.py::compute_total_reward) ---
    reward_weight_velocity: float = 0.10
    reward_weight_angular_rate: float = 0.02
    reward_weight_energy: float = 0.0005
    reward_weight_crash: float = 10.0
    # Extra squared penalty on lateral drift + tilt, active ONLY during the
    # takeoff phase -- enforces "straight up, no roll/pitch/yaw" per the
    # user's confirmed requirement (rewards.py::takeoff_stability_penalty).
    reward_weight_takeoff_stability: float = 0.50

    # --- domain randomization (added 2026-08-18, see domain_randomization.py
    # module docstring for the full sim-to-real rationale) ---
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    def for_evaluation(self) -> "DroneTaskEnvCfg":
        """Return this same cfg with all domain randomization disabled --
        use for scripts/sb3/play.py or any other clean/deterministic
        rollout. Does not mutate self.
        """
        self.domain_randomization = DomainRandomizationCfg.all_disabled()
        return self


@configclass
class DroneHoverEnvCfg(DroneTaskEnvCfg):
    """Hover task: after takeoff, hold position at the fixed target
    (0, 0, hover_height_m), rewarding near-zero linear/angular velocity.
    Registered as gym id Isaac-DroneHover-Direct-v0.
    """

    task_mode: str = TASK_HOVER


@configclass
class DroneFigure8EnvCfg(DroneTaskEnvCfg):
    """Figure-8 task: after takeoff, track a lemniscate
    (X = A*sin(t), Y = B*sin(t)*cos(t)) at constant hover_height_m.
    A/B/period are defined in task_targets.py (single source of truth) so
    training and any later inference-time target generator stay in sync.
    Registered as gym id Isaac-DroneFigure8-Direct-v0.
    """

    task_mode: str = TASK_FIGURE8
    episode_length_s: float = FIGURE8_PERIOD_S * 3.0  # a few full loops per episode


@configclass
class DroneShuttleRunEnvCfg(DroneTaskEnvCfg):
    """Shuttle-run task: after takeoff, the target alternates between
    X=-2.5m and X=+2.5m every SHUTTLE_RUN_SWITCH_PERIOD_S seconds (Y=0,
    constant hover_height_m) -- an aggressive accelerate/brake drill.
    Registered as gym id Isaac-DroneShuttleRun-Direct-v0.
    """

    task_mode: str = TASK_SHUTTLE_RUN
    episode_length_s: float = SHUTTLE_RUN_SWITCH_PERIOD_S * 6.0  # a few full legs per episode
