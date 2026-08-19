"""Train one of the three drone tasks with Stable-Baselines3 PPO.

Follows the AppLauncher + --headless pattern documented at
https://isaac-sim.github.io/IsaacLab/main/source/how-to/wrap_rl_env.html and
https://isaac-sim.github.io/IsaacLab/main/source/how-to/configuring_rl_training.html

Three tasks are registered (see tasks/direct/drone_navigation/__init__.py),
all sharing the same ground-idle -> vertical-ascent takeoff phase and
RC-style [roll,pitch,yaw,throttle] control pipeline -- only the
post-takeoff target differs:
    Isaac-DroneHover-Direct-v0       hold position at room center, 1m up
    Isaac-DroneFigure8-Direct-v0     track a lemniscate at constant height
    Isaac-DroneShuttleRun-Direct-v0  bounce between +/-2.0m on X

Usage:
    python scripts/sb3/train.py --task Isaac-DroneHover-Direct-v0 \
        --num_envs 4096 --headless

Run `python scripts/sb3/train.py --help` for all AppLauncher/CLI flags
(--headless, --num_envs, --seed, --max_iterations, etc. are provided by
Isaac Lab's own CLI argument helpers, not redefined here).
"""

import argparse

from isaaclab.app import AppLauncher

# --------------------------------------------------------------------------
# CLI args -- parsed BEFORE launching the Omniverse app, per Isaac Lab
# convention (AppLauncher needs to consume its own subset of argv first).
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train the drone navigation task with SB3 PPO.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-DroneHover-Direct-v0",
    choices=[
        "Isaac-DroneHover-Direct-v0",
        "Isaac-DroneFigure8-Direct-v0",
        "Isaac-DroneShuttleRun-Direct-v0",
    ],
)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None, help="Override n_timesteps.")
parser.add_argument(
    "--log_root",
    type=str,
    default="logs/sb3",
    help="Root directory for checkpoints and Tensorboard logs.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------
# Everything below imports isaaclab / isaacsim modules, which are only
# importable AFTER the AppLauncher has started the Omniverse app above.
# --------------------------------------------------------------------------

import os
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry

import autonomous_pipeline.tasks  # noqa: F401  (registers all 3 Isaac-Drone*-Direct-v0 ids)


def main() -> None:
    env_cfg: DirectRLEnvCfg = parse_env_cfg(
        args_cli.task, device=app_launcher.device, num_envs=args_cli.num_envs
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    sb3_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")
    agent_cfg = process_sb3_cfg(sb3_cfg, num_envs=env.unwrapped.num_envs)


    n_timesteps = agent_cfg.pop("n_timesteps")
    if args_cli.max_iterations is not None:
        n_timesteps = args_cli.max_iterations

    if agent_cfg.pop("normalize_input", False):
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=agent_cfg.pop("normalize_value", False),
            clip_obs=agent_cfg.pop("clip_obs", 10.0),
            gamma=agent_cfg.get("gamma", 0.99),
        )

    run_dir = os.path.join(
        args_cli.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(run_dir, exist_ok=True)

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed

    model = PPO(env=env, tensorboard_log=run_dir, verbose=1, **agent_cfg)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // max(args_cli.num_envs, 1),
        save_path=run_dir,
        name_prefix="model",
    )

    model.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    model.save(os.path.join(run_dir, "model_final"))
    if isinstance(env, VecNormalize):
        env.save(os.path.join(run_dir, "vecnormalize.pkl"))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
