"""Print every gym environment ID this project has registered.

Sanity-check script: run this first after any Isaac Lab install/environment
change to confirm all three drone task ids (`Isaac-DroneHover-Direct-v0`,
`Isaac-DroneFigure8-Direct-v0`, `Isaac-DroneShuttleRun-Direct-v0`) are
discoverable before attempting a full training run.

Usage: python scripts/list_envs.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List registered gym environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import autonomous_pipeline.tasks  # noqa: F401


def main() -> None:
    project_envs = [
        env_id for env_id in gym.envs.registry.keys() if "DroneNavigation" in env_id
    ]
    print("Registered autonomous_pipeline environments:")
    for env_id in project_envs:
        print(f"  - {env_id}")
    if not project_envs:
        print("  (none found -- check that autonomous_pipeline.tasks imported without error)")


if __name__ == "__main__":
    main()
    simulation_app.close()
