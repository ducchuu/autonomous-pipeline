"""Agent (learning-library) configs for the drone_navigation task.

Currently only Stable-Baselines3 PPO is configured (sb3_ppo_cfg.yaml). Add
sibling files here (e.g. rsl_rl_ppo_cfg.py, skrl_ppo_cfg.yaml) if you later
want to compare frameworks -- register each in the sibling __init__.py's
gym.register kwargs under its own `<framework>_cfg_entry_point` key, mirroring
the isaaclab_tasks convention.
"""
