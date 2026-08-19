"""Pure-Python unit tests for battery.py -- no isaaclab/isaacsim import.

Run with: pytest tests/test_battery.py -v
"""

import importlib.util
from pathlib import Path

import pytest
import torch

# Load battery.py directly by file path, bypassing the parent package's
# __init__ chain (which imports isaaclab/gymnasium via gym.register in
# drone_navigation/__init__.py). battery.py itself only depends on `torch`,
# so this keeps the test runnable without any Isaac Lab install, matching
# the module's own docstring claim.
_BATTERY_PATH = (
    Path(__file__).resolve().parents[1]
    / "source" / "autonomous_pipeline" / "autonomous_pipeline"
    / "tasks" / "direct" / "drone_navigation" / "battery.py"
)
_spec = importlib.util.spec_from_file_location("drone_battery", _BATTERY_PATH)
_battery_module = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules[_spec.name] = _battery_module  # dataclass() needs the module pre-registered
_spec.loader.exec_module(_battery_module)

BatteryModel = _battery_module.BatteryModel
CELL_COUNT = _battery_module.CELL_COUNT
CUTOFF_V_PER_CELL = _battery_module.CUTOFF_V_PER_CELL
FULL_CHARGE_V_PER_CELL = _battery_module.FULL_CHARGE_V_PER_CELL


@pytest.fixture
def battery():
    return BatteryModel(num_envs=3, device=torch.device("cpu"))


def test_starts_fully_charged(battery):
    assert torch.allclose(battery.state_of_charge(), torch.ones(3))


def test_open_circuit_voltage_at_full_charge_matches_spec(battery):
    ocv = battery.open_circuit_voltage()
    expected = CELL_COUNT * FULL_CHARGE_V_PER_CELL
    assert torch.allclose(ocv, torch.full((3,), expected), atol=1e-3)


def test_step_reduces_soc_under_nonzero_power(battery):
    soc_before = battery.state_of_charge().clone()
    battery.step(electrical_power_w=torch.full((3,), 50.0), dt_s=1.0)
    soc_after = battery.state_of_charge()
    assert torch.all(soc_after < soc_before)


def test_zero_power_draw_does_not_change_soc(battery):
    soc_before = battery.state_of_charge().clone()
    battery.step(electrical_power_w=torch.zeros(3), dt_s=1.0)
    soc_after = battery.state_of_charge()
    assert torch.allclose(soc_before, soc_after)


def test_terminal_voltage_sags_below_ocv_under_load(battery):
    ocv_before = battery.open_circuit_voltage()
    terminal_v = battery.step(electrical_power_w=torch.full((3,), 200.0), dt_s=0.02)
    assert torch.all(terminal_v < ocv_before)


def test_is_dead_when_voltage_below_cutoff(battery):
    cutoff = CELL_COUNT * CUTOFF_V_PER_CELL
    below_cutoff = torch.full((3,), cutoff - 0.5)
    assert torch.all(battery.is_dead(below_cutoff))


def test_is_dead_false_at_full_charge_no_load(battery):
    ocv = battery.open_circuit_voltage()
    assert not torch.any(battery.is_dead(ocv))


def test_reset_restores_full_charge_for_selected_envs(battery):
    battery.step(electrical_power_w=torch.full((3,), 100.0), dt_s=60.0)
    soc_after_drain = battery.state_of_charge()
    assert torch.all(soc_after_drain < 1.0)

    battery.reset(env_ids=torch.tensor([0, 1]))
    soc_after_reset = battery.state_of_charge()
    assert soc_after_reset[0] == pytest.approx(1.0)
    assert soc_after_reset[1] == pytest.approx(1.0)
    # env 2 was not reset, should still be drained.
    assert soc_after_reset[2] == pytest.approx(soc_after_drain[2].item())


def test_prolonged_high_draw_eventually_dies(battery):
    terminal_v = battery.open_circuit_voltage()
    for _ in range(2000):
        terminal_v = battery.step(electrical_power_w=torch.full((3,), 300.0), dt_s=1.0)
        if torch.all(battery.is_dead(terminal_v)):
            break
    assert torch.all(battery.is_dead(terminal_v))
