"""Reward terms for the drone navigation task.

See DRONE_SPEC.md section 5 for the full mathematical definition and the
rationale for each term. Kept as small, independently testable pure
functions operating on torch tensors of shape (num_envs, ...).
"""

from __future__ import annotations

import torch


def position_tracking_reward(pos_error_norm: torch.Tensor) -> torch.Tensor:
    """1 / (1 + ||p - p_target||). Bounded in (0, 1], maximal at the target.

    pos_error_norm: (num_envs,) euclidean distance to target, meters.
    """
    return 1.0 / (1.0 + pos_error_norm)


def velocity_damping_reward(lin_vel_norm: torch.Tensor) -> torch.Tensor:
    """exp(-||v||): rewards low speed, most impactful for the agent once it
    is already near the target (encourages settling, not overshoot).

    lin_vel_norm: (num_envs,) linear velocity magnitude, m/s.
    """
    return torch.exp(-lin_vel_norm)


def angular_rate_penalty(ang_vel_norm_sq: torch.Tensor) -> torch.Tensor:
    """||omega||^2 penalty -- discourages tumbling / oscillation.

    ang_vel_norm_sq: (num_envs,) squared angular velocity magnitude, (rad/s)^2.
    """
    return ang_vel_norm_sq


def energy_efficiency_penalty(electrical_power_w: torch.Tensor) -> torch.Tensor:
    """Instantaneous electrical power draw (watts). Penalizing the
    instantaneous value (not cumulative energy) avoids incentivizing the
    agent to front-load battery spend early in the episode.

    electrical_power_w: (num_envs,) watts, from physics.induced_electrical_power.
    """
    return electrical_power_w


def crash_penalty(crashed: torch.Tensor) -> torch.Tensor:
    """1.0 where crashed this step, else 0.0. crashed: bool tensor (num_envs,)."""
    return crashed.float()


def takeoff_stability_penalty(
    lateral_pos_error_norm: torch.Tensor, tilt_rad: torch.Tensor
) -> torch.Tensor:
    """Extra penalty applied ONLY during the takeoff phase (see
    flight_phase.py): the confirmed requirement is that the ascent be
    straight up, with X/Y stable and no roll/pitch/yaw. The generic
    position_tracking_reward already pulls X/Y/Z together, but this term
    adds a sharper, squared penalty specifically on lateral drift and tilt
    so the policy doesn't "cheat" a fast ascent by drifting sideways or
    tipping over, since those would still score well under a pure
    distance-to-target metric if the target is directly overhead.

    lateral_pos_error_norm: (num_envs,) sqrt(dx^2 + dy^2) from the ascent
        waypoint, meters.
    tilt_rad: (num_envs,) angle between body +Z and world +Z, radians.
    """
    return lateral_pos_error_norm**2 + tilt_rad**2


def compute_total_reward(
    pos_error_norm: torch.Tensor,
    lin_vel_norm: torch.Tensor,
    ang_vel_norm_sq: torch.Tensor,
    electrical_power_w: torch.Tensor,
    crashed: torch.Tensor,
    weight_velocity: float,
    weight_angular_rate: float,
    weight_energy: float,
    weight_crash: float,
    is_takeoff_phase: torch.Tensor | None = None,
    lateral_pos_error_norm: torch.Tensor | None = None,
    tilt_rad: torch.Tensor | None = None,
    weight_takeoff_stability: float = 0.0,
) -> torch.Tensor:
    """Combine all terms per DRONE_SPEC.md section 5. Weights come from
    DroneTaskEnvCfg so they are tunable without touching this file.

    The takeoff-phase args are optional and default to a no-op so this
    function still works for callers/tests that don't model the takeoff
    state machine; DroneTaskEnv (drone_navigation_env.py) always passes them.
    """
    reward = position_tracking_reward(pos_error_norm)
    reward = reward + weight_velocity * velocity_damping_reward(lin_vel_norm)
    reward = reward - weight_angular_rate * angular_rate_penalty(ang_vel_norm_sq)
    reward = reward - weight_energy * energy_efficiency_penalty(electrical_power_w)
    reward = reward - weight_crash * crash_penalty(crashed)

    if is_takeoff_phase is not None and weight_takeoff_stability > 0.0:
        stability_penalty = takeoff_stability_penalty(lateral_pos_error_norm, tilt_rad)
        reward = reward - weight_takeoff_stability * is_takeoff_phase.float() * stability_penalty

    return reward
