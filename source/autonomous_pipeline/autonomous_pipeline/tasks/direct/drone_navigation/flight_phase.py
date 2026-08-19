"""Takeoff -> task phase state machine, shared by all three tasks.

Confirmed requirement (user, 2026-08-18): every task starts with the drone
armed and idle, resting on the ground at the room center. It must first
ascend on Z only (X/Y stationary, zero roll/pitch/yaw) to HOVER_HEIGHT_M and
hold there briefly before the task-specific target (hover/shuttle-run/
figure-8, from task_targets.py) takes over.

This module tracks that per-environment phase transition with plain torch
tensors -- no isaaclab dependency, unit-testable in isolation (see
tests/test_flight_phase.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .task_targets import HOVER_HEIGHT_M

PHASE_TAKEOFF = 0
PHASE_TASK = 1

# Stable-hover checkpoint tolerances -- [TUNABLE-TRAINING], not physical
# constants. Loosen these if the policy struggles to ever trigger the
# transition; tighten them if the task phase starts from too rough a state.
POSITION_TOLERANCE_M = 0.15
VELOCITY_TOLERANCE_MPS = 0.20
TILT_TOLERANCE_RAD = 0.0873  # ~5 degrees
DWELL_TIME_S = 0.5  # must satisfy all tolerances continuously for this long


@dataclass
class FlightPhaseState:
    """Per-environment phase-tracking tensors. All shape (num_envs,)."""

    phase: torch.Tensor       # int64, PHASE_TAKEOFF or PHASE_TASK
    dwell_timer_s: torch.Tensor  # float, time spent continuously within tolerance
    task_clock_s: torch.Tensor   # float, time since entering PHASE_TASK

    @classmethod
    def create(cls, num_envs: int, device: torch.device) -> "FlightPhaseState":
        return cls(
            phase=torch.full((num_envs,), PHASE_TAKEOFF, dtype=torch.int64, device=device),
            dwell_timer_s=torch.zeros(num_envs, device=device),
            task_clock_s=torch.zeros(num_envs, device=device),
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self.phase[env_ids] = PHASE_TAKEOFF
        self.dwell_timer_s[env_ids] = 0.0
        self.task_clock_s[env_ids] = 0.0


def takeoff_target(num_envs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The single ascent waypoint for the takeoff phase: directly above the
    ground-idle spawn point, at HOVER_HEIGHT_M. Same (x=0, y=0) as the hover
    task's target by construction -- ascent is a pure Z-axis motion.
    """
    target = torch.tensor((0.0, 0.0, HOVER_HEIGHT_M), device=device, dtype=dtype)
    return target.unsqueeze(0).expand(num_envs, -1)


def update(
    state: FlightPhaseState,
    pos_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    tilt_rad: torch.Tensor,
    dt_s: float,
) -> None:
    """Advance the phase state machine by one control step, in place.

    pos_w:     (num_envs, 3) world position
    lin_vel_w: (num_envs, 3) world linear velocity
    tilt_rad:  (num_envs,) angle between body +Z and world +Z (0 = level)
    dt_s:      control step duration, seconds
    """
    in_takeoff = state.phase == PHASE_TAKEOFF

    ascent_target = takeoff_target(pos_w.shape[0], pos_w.device, pos_w.dtype)
    pos_error = torch.linalg.norm(ascent_target - pos_w, dim=-1)
    vel_norm = torch.linalg.norm(lin_vel_w, dim=-1)

    within_tolerance = (
        (pos_error < POSITION_TOLERANCE_M)
        & (vel_norm < VELOCITY_TOLERANCE_MPS)
        & (tilt_rad < TILT_TOLERANCE_RAD)
    )

    # Only accumulate/decay the dwell timer for envs still in takeoff.
    state.dwell_timer_s = torch.where(
        in_takeoff & within_tolerance,
        state.dwell_timer_s + dt_s,
        torch.where(in_takeoff, torch.zeros_like(state.dwell_timer_s), state.dwell_timer_s),
    )

    just_qualified = in_takeoff & (state.dwell_timer_s >= DWELL_TIME_S)
    state.phase = torch.where(
        just_qualified, torch.full_like(state.phase, PHASE_TASK), state.phase
    )

    in_task = state.phase == PHASE_TASK
    # Advance the task clock only for envs already in the task phase;
    # envs that just transitioned this step start their task clock at 0.
    state.task_clock_s = torch.where(
        in_task & ~just_qualified,
        state.task_clock_s + dt_s,
        torch.where(just_qualified, torch.zeros_like(state.task_clock_s), state.task_clock_s),
    )


def active_target(
    state: FlightPhaseState, task_target_fn
) -> torch.Tensor:
    """Return the currently active target per environment: the fixed
    ascent waypoint while in PHASE_TAKEOFF, or the task-specific generator's
    output (evaluated at each env's own task_clock_s) once in PHASE_TASK.
    """
    num_envs = state.phase.shape[0]
    device = state.phase.device
    dtype = state.task_clock_s.dtype

    takeoff_pt = takeoff_target(num_envs, device, dtype)
    task_pt = task_target_fn(state.task_clock_s)

    in_task = (state.phase == PHASE_TASK).unsqueeze(-1)
    return torch.where(in_task, task_pt, takeoff_pt)
