"""Pure-Python unit tests for physics.py -- no isaaclab/isaacsim import, so
these run anywhere with `pip install torch pytest` (e.g. in CI, or before
Isaac Sim is even installed).

Run with: pytest tests/test_physics.py -v
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

# Load physics.py directly by file path instead of `import
# autonomous_pipeline...` -- importing through the real package would run
# every parent __init__.py in the chain (autonomous_pipeline ->
# tasks -> direct -> drone_navigation), and drone_navigation/__init__.py
# calls gym.register() and imports isaaclab, which is NOT installed in this
# plain test environment. physics.py itself has zero isaaclab dependency
# (only `torch`), so loading it standalone keeps these tests runnable
# anywhere, exactly as documented in its module docstring.
_PHYSICS_PATH = (
    Path(__file__).resolve().parents[1]
    / "source" / "autonomous_pipeline" / "autonomous_pipeline"
    / "tasks" / "direct" / "drone_navigation" / "physics.py"
)
_spec = importlib.util.spec_from_file_location("drone_physics", _PHYSICS_PATH)
physics = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules[_spec.name] = physics  # dataclass() needs the module pre-registered
_spec.loader.exec_module(physics)


def test_mass_components_sum_to_total():
    total = physics.MASS_CORE_KG + physics.MASS_MOTORS_TOTAL_KG + physics.MASS_BATTERY_KG
    assert math.isclose(total, physics.MASS_TOTAL_KG, rel_tol=1e-9)


def test_thrust_to_weight_ratio_matches_measured_spec():
    total_max_thrust_n = physics.MAX_THRUST_PER_MOTOR_N * physics.N_MOTORS
    weight_n = physics.MASS_TOTAL_KG * physics.GRAVITY_MPS2
    twr = total_max_thrust_n / weight_n
    # DRONE_SPEC.md documents 11.07:1 -- keep this test in sync if you ever
    # correct MAX_THRUST_PER_MOTOR_GF_AT_4S or MASS_TOTAL_KG.
    assert twr == pytest.approx(11.07, abs=0.01)


def test_motor_layout_has_four_symmetric_arms():
    layout = physics.motor_layout()
    assert len(layout.positions_xy_m) == 4
    for x, y in layout.positions_xy_m:
        r = math.hypot(x, y)
        assert r == pytest.approx(physics.ARM_RADIUS_M, rel=1e-4)


def test_mass_properties_com_height_between_frame_and_battery_top():
    props = physics.compute_mass_properties()
    assert props.total_mass_kg == pytest.approx(physics.MASS_TOTAL_KG)
    # CoM must land somewhere between the frame bottom (0) and the battery's
    # top surface -- a basic physical sanity bound, independent of the exact
    # [ESTIMATE-VERIFY] geometry constants.
    battery_top = physics.FRAME_HEIGHT_M + 2 * physics.BATTERY_HALF_THICKNESS_M
    assert 0.0 < props.com_height_m < battery_top


def test_inertia_tensor_is_positive_and_izz_exceeds_ixx_iyy_for_flat_wide_frame():
    props = physics.compute_mass_properties()
    ixx, iyy, izz = props.inertia_diag_kgm2
    assert ixx > 0 and iyy > 0 and izz > 0
    # A flat, wide X-quad (motors far out in the xy-plane, short in z) should
    # have more yaw inertia (Izz) than roll/pitch inertia (Ixx/Iyy), because
    # the motor point masses contribute their full arm-radius^2 to Izz but
    # only their much smaller z-offset^2 to Ixx/Iyy.
    assert izz > ixx
    assert izz > iyy


def test_thrust_from_action_bounds():
    actions = torch.tensor([[-1.0, 0.0, 1.0, -1.0]])
    thrusts = physics.thrust_from_action(actions)
    assert thrusts[0, 0] == pytest.approx(0.0)
    assert thrusts[0, 1] == pytest.approx(physics.MAX_THRUST_PER_MOTOR_N / 2, rel=1e-4)
    assert thrusts[0, 2] == pytest.approx(physics.MAX_THRUST_PER_MOTOR_N, rel=1e-4)


def test_mixer_hover_thrust_produces_zero_net_torque_when_equal():
    equal_thrust = physics.MASS_TOTAL_KG * physics.GRAVITY_MPS2 / physics.N_MOTORS
    thrusts = torch.full((1, 4), equal_thrust)
    forces, torques = physics.mixer_forces_and_torques(thrusts)
    assert forces[0, 2] == pytest.approx(physics.MASS_TOTAL_KG * physics.GRAVITY_MPS2, rel=1e-4)
    # Roll/pitch torques cancel by symmetry when all 4 thrusts are equal.
    assert torques[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert torques[0, 1] == pytest.approx(0.0, abs=1e-6)
    # Yaw torque also cancels: two CCW and two CW motors at equal thrust.
    assert torques[0, 2] == pytest.approx(0.0, abs=1e-6)


def test_induced_power_increases_with_thrust():
    low = physics.induced_electrical_power(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    high = physics.induced_electrical_power(torch.tensor([[5.0, 5.0, 5.0, 5.0]]))
    assert high.item() > low.item()


def test_induced_power_is_zero_at_zero_thrust():
    power = physics.induced_electrical_power(torch.zeros((1, 4)))
    assert power.item() == pytest.approx(0.0)
