"""Gym registration for the drone_navigation task family.

Three separate gym IDs share the SAME env class (DroneTaskEnv) and differ
only by which config subclass they register -- task_mode on the config is
what actually selects hover / figure-8 / shuttle-run behavior at runtime
(see drone_navigation_env_cfg.py and task_targets.py). This mirrors how the
takeoff phase, control pipeline, and observation vector are intentionally
identical across all three tasks.

Follows the registration pattern documented at
https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html
"""

import gymnasium as gym

from . import agents
from .drone_navigation_env import DroneTaskEnv
from .drone_navigation_env_cfg import DroneFigure8EnvCfg, DroneHoverEnvCfg, DroneShuttleRunEnvCfg

gym.register(
    id="Isaac-DroneHover-Direct-v0",
    entry_point=f"{__name__}.drone_navigation_env:DroneTaskEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_navigation_env_cfg:DroneHoverEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DroneFigure8-Direct-v0",
    entry_point=f"{__name__}.drone_navigation_env:DroneTaskEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_navigation_env_cfg:DroneFigure8EnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DroneShuttleRun-Direct-v0",
    entry_point=f"{__name__}.drone_navigation_env:DroneTaskEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_navigation_env_cfg:DroneShuttleRunEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
