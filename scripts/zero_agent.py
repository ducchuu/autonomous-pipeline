"""Run the drone navigation env with an all-zero action every step.

Sanity check #2 (after list_envs.py): confirms the scene loads, the drone
spawns, physics steps without erroring, and the drone simply falls under
gravity with zero thrust (a good visual check that gravity + mass are wired
correctly, before any policy is involved).

Usage: python scripts/zero_agent.py --num_envs 4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-action smoke test.")
parser.add_argument("--task", type=str, default="Isaac-DroneHover-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

import autonomous_pipeline.tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=app_launcher.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    env.reset()
    actions = torch.full((args_cli.num_envs, 4), -1.0, device=env.unwrapped.device)
    while simulation_app.is_running():
        env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
