"""Run the drone navigation env with random actions every step.

Sanity check #3: confirms the full observation/action/reward/reset loop runs
end-to-end without shape errors, before spending time on real training.

Usage: python scripts/random_agent.py --num_envs 16
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random-action smoke test.")
parser.add_argument("--task", type=str, default="Isaac-DroneHover-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

import autonomous_pipeline.tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=app_launcher.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    env.reset()
    while simulation_app.is_running():
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
