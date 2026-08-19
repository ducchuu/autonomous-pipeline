"""Pure-Python unit tests for task_targets.py.

Run with: pytest tests/test_task_targets.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PATH = (
    Path(__file__).resolve().parents[1]
    / "source" / "autonomous_pipeline" / "autonomous_pipeline"
    / "tasks" / "direct" / "drone_navigation" / "task_targets.py"
)
_spec = importlib.util.spec_from_file_location("drone_task_targets", _PATH)
task_targets = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = task_targets
_spec.loader.exec_module(task_targets)


def test_hover_target_is_fixed_point():
    t = torch.tensor([0.0, 3.7, 100.0])
    target = task_targets.hover_target(t)
    expected = torch.tensor(task_targets.HOVER_TARGET_XYZ)
    for i in range(3):
        assert torch.allclose(target[i], expected)


def test_shuttle_run_starts_at_negative_x():
    t = torch.tensor([0.0])
    target = task_targets.shuttle_run_target(t)
    assert target[0, 0].item() == pytest.approx(-task_targets.SHUTTLE_RUN_HALF_LENGTH_M)
    assert target[0, 2].item() == pytest.approx(task_targets.HOVER_HEIGHT_M)


def test_shuttle_run_flips_after_switch_period():
    period = task_targets.SHUTTLE_RUN_SWITCH_PERIOD_S
    t = torch.tensor([0.0, period * 0.5, period * 1.5, period * 2.5])
    target = task_targets.shuttle_run_target(t)
    xs = target[:, 0]
    assert xs[0].item() == pytest.approx(-task_targets.SHUTTLE_RUN_HALF_LENGTH_M)
    assert xs[1].item() == pytest.approx(-task_targets.SHUTTLE_RUN_HALF_LENGTH_M)
    assert xs[2].item() == pytest.approx(task_targets.SHUTTLE_RUN_HALF_LENGTH_M)
    assert xs[3].item() == pytest.approx(-task_targets.SHUTTLE_RUN_HALF_LENGTH_M)


def test_figure8_starts_at_origin_and_stays_flat():
    t = torch.tensor([0.0, 1.0, 5.0])
    target = task_targets.figure8_target(t)
    assert target[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert target[0, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(target[:, 2], torch.full((3,), task_targets.HOVER_HEIGHT_M))


def test_figure8_bounded_by_confirmed_amplitudes():
    t = torch.linspace(0.0, task_targets.FIGURE8_PERIOD_S * 2, 500)
    target = task_targets.figure8_target(t)
    assert target[:, 0].abs().max().item() <= task_targets.FIGURE8_A_M + 1e-6
    assert target[:, 1].abs().max().item() <= task_targets.FIGURE8_B_M + 1e-6


def test_get_task_target_fn_returns_correct_function():
    assert task_targets.get_task_target_fn(task_targets.TASK_HOVER) is task_targets.hover_target
    assert (
        task_targets.get_task_target_fn(task_targets.TASK_SHUTTLE_RUN)
        is task_targets.shuttle_run_target
    )
    assert (
        task_targets.get_task_target_fn(task_targets.TASK_FIGURE8) is task_targets.figure8_target
    )


def test_get_task_target_fn_raises_on_unknown_task():
    with pytest.raises(ValueError):
        task_targets.get_task_target_fn("not_a_real_task")


# --- cage-margin safety cross-checks (added 2026-08-19, cage rescaled to
# 8m x 5m x 2.5m -- see DRONE_SPEC.md section 8/10). These intentionally
# hardcode the world_bounds_x_m/_y_m values from
# drone_navigation_env_cfg.py::DroneTaskEnvCfg (4.0 / 2.5) because that file
# imports isaaclab and cannot be loaded in this isaaclab-free test suite --
# if you change either the cage bounds or these task amplitudes, update both
# this file and drone_navigation_env_cfg.py together, and re-check the
# margin below still holds.
_WORLD_BOUND_X_M = 4.0
_WORLD_BOUND_Y_M = 2.5
_MIN_SAFETY_MARGIN_M = 1.0  # minimum clearance we require between a task's
# peak waypoint displacement and the crash/termination boundary on that axis


def test_shuttle_run_half_length_clears_world_bound_x_with_margin():
    clearance = _WORLD_BOUND_X_M - task_targets.SHUTTLE_RUN_HALF_LENGTH_M
    assert clearance >= _MIN_SAFETY_MARGIN_M


def test_figure8_peak_x_clears_world_bound_x_with_margin():
    peak_x_m = task_targets.FIGURE8_A_M  # X(t) = A*sin(wt), peak = A
    clearance = _WORLD_BOUND_X_M - peak_x_m
    assert clearance >= _MIN_SAFETY_MARGIN_M


def test_figure8_peak_y_clears_world_bound_y_with_margin():
    # Y(t) = B*sin(wt)*cos(wt) = (B/2)*sin(2wt) -- peak is B/2, NOT B (see
    # the doc-bug note in DRONE_SPEC.md section 10 / task_targets.py).
    peak_y_m = task_targets.FIGURE8_B_M / 2.0
    clearance = _WORLD_BOUND_Y_M - peak_y_m
    assert clearance >= _MIN_SAFETY_MARGIN_M
