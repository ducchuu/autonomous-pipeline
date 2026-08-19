"""Play/evaluate a trained SB3 PPO checkpoint on one of the three drone
tasks (Isaac-DroneHover-Direct-v0 / Isaac-DroneFigure8-Direct-v0 /
Isaac-DroneShuttleRun-Direct-v0 -- pass whichever one matches the
checkpoint's training run via --task).

Usage:
    python scripts/sb3/play.py --task Isaac-DroneHover-Direct-v0 \
        --checkpoint logs/sb3/<run>/model_final.zip --num_envs 16

Omit --headless (default off here) to watch it fly in the Isaac Sim GUI.
Per DRONE_SPEC.md section 7, expect GUI/RTX-rendered playback to be
noticeably slower than headless training on an RTX A4000 -- this is
expected, not a bug.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained drone navigation policy.")
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
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .zip file.")
parser.add_argument(
    "--vecnormalize_path",
    type=str,
    default=None,
    help="Optional path to a saved VecNormalize .pkl from the matching training run.",
)
parser.add_argument(
    "--domain-randomization",
    dest="domain_randomization",
    action="store_true",
    help="Keep domain randomization (action latency, mass/CoM, motor efficiency, "
    "observation noise, wind gusts) enabled during playback. Default is DISABLED "
    "for play.py, so you see the policy's clean behavior for debugging/visual "
    "inspection; re-enable this only if you specifically want to sanity-check "
    "robustness to randomization during a watched rollout.",
)
parser.set_defaults(domain_randomization=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import autonomous_pipeline.tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=app_launcher.device, num_envs=args_cli.num_envs)

    # Domain randomization is ON by default in every DroneTaskEnvCfg (it must
    # be, for training) -- disable it here for a clean playback/debugging
    # rollout unless the user explicitly asked to keep it via --domain-randomization.
    # See DomainRandomizationCfg / DroneTaskEnvCfg.for_evaluation() in
    # drone_navigation_env_cfg.py.
    if not args_cli.domain_randomization:
        env_cfg.for_evaluation()

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = Sb3VecEnvWrapper(env)

    if args_cli.vecnormalize_path is not None:
        env = VecNormalize.load(args_cli.vecnormalize_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args_cli.checkpoint, env=env)

    obs = env.reset()
    while simulation_app.is_running():
        actions, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
