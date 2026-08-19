"""Task-specific target generators for the drone navigation task family.

All three tasks (hover, shuttle-run, figure-8) share the SAME takeoff
sequence (see flight_phase.py): the drone starts armed and idle on the
ground, ascends straight up on Z only to HOVER_HEIGHT_M with roll/pitch/yaw
locked near zero, and only once that stable-hover checkpoint is reached does
the task-specific target below start driving the drone.

Per the confirmed sim-to-real architecture, these targets are exactly the
"virtual waypoint" the real inference-time script will also generate and
feed into the same observation slot -- nothing here is a physical object in
the world (no literal gates yet; see DRONE_SPEC.md section 9 for the planned
future physical-gate extension).

All constants below are TRAINING/TASK-DESIGN parameters (confirmed with the
user 2026-08-18), not physical unknowns -- but SWITCH_PERIOD_S and
FIGURE8_PERIOD_S are explicitly flagged as tunable: they set how aggressively
the drone must accelerate/decelerate, and the right value depends on this
drone's actual achievable acceleration, which you'll only know from training
curves (and eventually flight tests).

No isaaclab dependency -- pure torch, unit-testable in isolation (see
tests/test_task_targets.py).
"""

from __future__ import annotations

import math

import torch

HOVER_HEIGHT_M = 1.0  # [CONFIRMED by user 2026-08-18] all three tasks fly flat at this height

# --- hover task ---
HOVER_TARGET_XYZ = (0.0, 0.0, HOVER_HEIGHT_M)  # [CONFIRMED] room center, 1m up

# --- shuttle-run task ---
# RESCALED 2026-08-19: OptiTrack cage re-measured at 8m(X) x 5m(Y) x 2.5m(Z)
# (see DRONE_SPEC.md section 8), superseding the earlier 10m x 10m figure.
# World-bound termination on X is now +/-4m (drone_navigation_env_cfg.py).
# 2.0m half-length keeps a 2.0m clearance to that bound on each end (same
# absolute clearance the previous 2.5m half-length had against the previous
# +/-5m bound), leaving room for brake/overshoot at the end of each leg.
SHUTTLE_RUN_HALF_LENGTH_M = 2.0   # [RESCALED 2026-08-19] X = -2.0 .. +2.0 (4m total)
SHUTTLE_RUN_SWITCH_PERIOD_S = 4.0  # [TUNABLE-TRAINING] seconds per leg before target flips

# --- figure-8 task ---
# RESCALED 2026-08-19 for the same re-measured 8m x 5m x 2.5m cage.
# NOTE (pre-existing doc bug, fixed here): the actual peak |Y| reached by
# Y(t) = B*sin(t)*cos(t) = (B/2)*sin(2t) is B/2, NOT B -- the old "Y
# half-range = B = 2.0m" comment overstated the real peak (which was
# always +/-1.0m). Left B numerically unchanged (peak Y = +/-1.0m) since
# that already clears the new, tighter Y world-bound (+/-2.5m) with 1.5m
# margin on each side. A is reduced to match X's new clearance budget
# (peak X = +/-2.0m against the new +/-4m X bound, 2.0m margin, same as
# the shuttle-run task above).
FIGURE8_A_M = 2.0   # [RESCALED 2026-08-19] X half-range, peak X = +/-2.0m
FIGURE8_B_M = 2.0   # [UNCHANGED 2026-08-19] peak Y = B/2 = +/-1.0m (see note above)
FIGURE8_PERIOD_S = 10.0  # [TUNABLE-TRAINING] seconds for one full figure-8 loop

TASK_HOVER = "hover"
TASK_SHUTTLE_RUN = "shuttle_run"
TASK_FIGURE8 = "figure8"
ALL_TASKS = (TASK_HOVER, TASK_SHUTTLE_RUN, TASK_FIGURE8)


def hover_target(task_time_s: torch.Tensor) -> torch.Tensor:
    """Fixed point at room center, HOVER_HEIGHT_M up. task_time_s is unused
    (kept for a uniform function signature across all three task generators)
    but its shape drives the output batch shape.

    task_time_s: (num_envs,) seconds since entering the task phase
    returns: (num_envs, 3) target position, world frame
    """
    target = torch.tensor(HOVER_TARGET_XYZ, device=task_time_s.device, dtype=task_time_s.dtype)
    return target.unsqueeze(0).expand(task_time_s.shape[0], -1)


def shuttle_run_target(task_time_s: torch.Tensor) -> torch.Tensor:
    """Alternates between (-HALF_LENGTH, 0, H) and (+HALF_LENGTH, 0, H) every
    SHUTTLE_RUN_SWITCH_PERIOD_S seconds. A pure time-based (not
    tolerance-based) flip keeps this vectorizable across many parallel
    envs without per-env branching logic.

    task_time_s: (num_envs,) seconds since entering the task phase
    returns: (num_envs, 3) target position, world frame
    """
    leg_index = torch.floor(task_time_s / SHUTTLE_RUN_SWITCH_PERIOD_S).long()
    sign = torch.where(leg_index % 2 == 0, -1.0, 1.0)
    x = sign * SHUTTLE_RUN_HALF_LENGTH_M
    y = torch.zeros_like(x)
    z = torch.full_like(x, HOVER_HEIGHT_M)
    return torch.stack([x, y, z], dim=-1)


def figure8_target(task_time_s: torch.Tensor) -> torch.Tensor:
    """Parametric lemniscate (figure-8) target, flat at HOVER_HEIGHT_M:

        X(t) = A * sin(w*t)
        Y(t) = B * sin(w*t) * cos(w*t)
        Z(t) = HOVER_HEIGHT_M

    with w = 2*pi / FIGURE8_PERIOD_S. See DRONE_SPEC.md section 9 for the
    rationale on keeping Z flat (isolates roll/pitch/yaw coordination from
    thrust/altitude compensation during early training).

    task_time_s: (num_envs,) seconds since entering the task phase
    returns: (num_envs, 3) target position, world frame
    """
    omega = 2.0 * math.pi / FIGURE8_PERIOD_S
    x = FIGURE8_A_M * torch.sin(omega * task_time_s)
    y = FIGURE8_B_M * torch.sin(omega * task_time_s) * torch.cos(omega * task_time_s)
    z = torch.full_like(x, HOVER_HEIGHT_M)
    return torch.stack([x, y, z], dim=-1)


_TASK_TARGET_FUNCTIONS = {
    TASK_HOVER: hover_target,
    TASK_SHUTTLE_RUN: shuttle_run_target,
    TASK_FIGURE8: figure8_target,
}


def get_task_target_fn(task_name: str):
    """Look up the target-generator function for a task name. Raises
    ValueError (not a silent None) on an unknown task name, so a config typo
    fails loudly at startup instead of producing a stationary drone.
    """
    if task_name not in _TASK_TARGET_FUNCTIONS:
        raise ValueError(
            f"Unknown task_name '{task_name}'. Must be one of {ALL_TASKS}."
        )
    return _TASK_TARGET_FUNCTIONS[task_name]
