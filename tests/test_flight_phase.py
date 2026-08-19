"""Pure-Python unit tests for flight_phase.py.

Run with: pytest tests/test_flight_phase.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_TASK_DIR = (
    Path(__file__).resolve().parents[1]
    / "source" / "autonomous_pipeline" / "autonomous_pipeline"
    / "tasks" / "direct" / "drone_navigation"
)

_task_targets_spec = importlib.util.spec_from_file_location(
    "drone_task_targets_for_phase_test", _TASK_DIR / "task_targets.py"
)
task_targets = importlib.util.module_from_spec(_task_targets_spec)
sys.modules[_task_targets_spec.name] = task_targets
_task_targets_spec.loader.exec_module(task_targets)

import types  # noqa: E402

_fake_pkg = types.ModuleType("drone_navigation_pkg_for_phase_test")
_fake_pkg.task_targets = task_targets
sys.modules["drone_navigation_pkg_for_phase_test"] = _fake_pkg
sys.modules["drone_navigation_pkg_for_phase_test.task_targets"] = task_targets

_phase_spec = importlib.util.spec_from_file_location(
    "drone_navigation_pkg_for_phase_test.flight_phase", _TASK_DIR / "flight_phase.py"
)
flight_phase = importlib.util.module_from_spec(_phase_spec)
flight_phase.__package__ = "drone_navigation_pkg_for_phase_test"
sys.modules[_phase_spec.name] = flight_phase
_phase_spec.loader.exec_module(flight_phase)


def test_starts_in_takeoff_phase():
    state = flight_phase.FlightPhaseState.create(num_envs=2, device=torch.device("cpu"))
    assert torch.all(state.phase == flight_phase.PHASE_TAKEOFF)


def test_stays_in_takeoff_when_far_from_target():
    state = flight_phase.FlightPhaseState.create(num_envs=1, device=torch.device("cpu"))
    pos = torch.tensor([[0.0, 0.0, 0.0]])  # still on the ground
    vel = torch.zeros((1, 3))
    tilt = torch.zeros(1)
    for _ in range(20):
        flight_phase.update(state, pos, vel, tilt, dt_s=0.02)
    assert state.phase[0].item() == flight_phase.PHASE_TAKEOFF


def test_transitions_to_task_after_dwell_time_at_target():
    state = flight_phase.FlightPhaseState.create(num_envs=1, device=torch.device("cpu"))
    pos = torch.tensor([[0.0, 0.0, task_targets.HOVER_HEIGHT_M]])
    vel = torch.zeros((1, 3))
    tilt = torch.zeros(1)
    dt = 0.02
    steps_needed = int(flight_phase.DWELL_TIME_S / dt) + 2
    for _ in range(steps_needed):
        flight_phase.update(state, pos, vel, tilt, dt_s=dt)
    assert state.phase[0].item() == flight_phase.PHASE_TASK


def test_dwell_timer_resets_if_it_drifts_out_of_tolerance():
    state = flight_phase.FlightPhaseState.create(num_envs=1, device=torch.device("cpu"))
    good_pos = torch.tensor([[0.0, 0.0, task_targets.HOVER_HEIGHT_M]])
    bad_pos = torch.tensor([[1.0, 1.0, task_targets.HOVER_HEIGHT_M]])
    vel = torch.zeros((1, 3))
    tilt = torch.zeros(1)
    dt = 0.02

    # Almost qualify...
    for _ in range(int(flight_phase.DWELL_TIME_S / dt) - 2):
        flight_phase.update(state, good_pos, vel, tilt, dt_s=dt)
    assert state.phase[0].item() == flight_phase.PHASE_TAKEOFF

    # ...then drift away, which must reset the dwell timer.
    flight_phase.update(state, bad_pos, vel, tilt, dt_s=dt)
    assert state.dwell_timer_s[0].item() == pytest.approx(0.0)


def test_task_clock_starts_at_zero_on_transition_and_advances_after():
    state = flight_phase.FlightPhaseState.create(num_envs=1, device=torch.device("cpu"))
    pos = torch.tensor([[0.0, 0.0, task_targets.HOVER_HEIGHT_M]])
    vel = torch.zeros((1, 3))
    tilt = torch.zeros(1)
    dt = 0.02
    steps_needed = int(flight_phase.DWELL_TIME_S / dt) + 2
    for _ in range(steps_needed):
        flight_phase.update(state, pos, vel, tilt, dt_s=dt)
    assert state.phase[0].item() == flight_phase.PHASE_TASK
    clock_at_transition = state.task_clock_s[0].item()
    flight_phase.update(state, pos, vel, tilt, dt_s=dt)
    assert state.task_clock_s[0].item() > clock_at_transition


def test_active_target_is_ascent_point_during_takeoff():
    state = flight_phase.FlightPhaseState.create(num_envs=1, device=torch.device("cpu"))
    target = flight_phase.active_target(state, task_targets.hover_target)
    expected = torch.tensor([[0.0, 0.0, task_targets.HOVER_HEIGHT_M]])
    assert torch.allclose(target, expected)


def test_reset_returns_to_takeoff_phase():
    state = flight_phase.FlightPhaseState.create(num_envs=2, device=torch.device("cpu"))
    state.phase[0] = flight_phase.PHASE_TASK
    state.task_clock_s[0] = 5.0
    state.reset(torch.tensor([0]))
    assert state.phase[0].item() == flight_phase.PHASE_TAKEOFF
    assert state.task_clock_s[0].item() == pytest.approx(0.0)
