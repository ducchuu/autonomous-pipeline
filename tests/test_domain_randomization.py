"""Pure-Python unit tests for domain_randomization.py.

Run with: pytest tests/test_domain_randomization.py -v
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

_TASK_DIR = (
    Path(__file__).resolve().parents[1]
    / "source" / "autonomous_pipeline" / "autonomous_pipeline"
    / "tasks" / "direct" / "drone_navigation"
)


def _load(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _TASK_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


physics = _load("drone_physics_for_dr_test", "physics.py")

# domain_randomization.py does `from . import physics` (relative import),
# same fake-parent-package trick used in test_attitude_controller.py.
_fake_pkg = types.ModuleType("drone_navigation_pkg_for_dr_test")
_fake_pkg.physics = physics
sys.modules["drone_navigation_pkg_for_dr_test"] = _fake_pkg
sys.modules["drone_navigation_pkg_for_dr_test.physics"] = physics

_spec = importlib.util.spec_from_file_location(
    "drone_navigation_pkg_for_dr_test.domain_randomization",
    _TASK_DIR / "domain_randomization.py",
)
dr = importlib.util.module_from_spec(_spec)
dr.__package__ = "drone_navigation_pkg_for_dr_test"
sys.modules[_spec.name] = dr
_spec.loader.exec_module(dr)


torch.manual_seed(0)


# --------------------------------------------------------------------------
# Mass scale
# --------------------------------------------------------------------------


def test_sample_mass_scale_within_range():
    scales = dr.sample_mass_scale(1000, torch.device("cpu"), torch.float32)
    low, high = dr.MASS_SCALE_RANGE
    assert torch.all(scales >= low)
    assert torch.all(scales <= high)


def test_apply_mass_scale_to_wrench_identity_at_scale_one():
    force = torch.tensor([[0.0, 0.0, 5.0]])
    torque = torch.tensor([[0.1, -0.2, 0.05]])
    scale = torch.tensor([1.0])
    out_force, out_torque = dr.apply_mass_scale_to_wrench(force, torque, scale)
    assert torch.allclose(out_force, force)
    assert torch.allclose(out_torque, torque)


def test_apply_mass_scale_to_wrench_heavier_drone_reduces_response():
    # A "heavier" emulated drone (scale > 1) should get LESS effective
    # force/torque per unit commanded, since 1/scale < 1.
    force = torch.tensor([[0.0, 0.0, 5.0]])
    torque = torch.tensor([[0.1, 0.0, 0.0]])
    scale = torch.tensor([1.2])
    out_force, out_torque = dr.apply_mass_scale_to_wrench(force, torque, scale)
    assert out_force[0, 2].item() < force[0, 2].item()
    assert out_torque[0, 0].item() < torque[0, 0].item()


# --------------------------------------------------------------------------
# CoM offset
# --------------------------------------------------------------------------


def test_sample_com_offset_xy_within_range():
    offsets = dr.sample_com_offset_xy(1000, torch.device("cpu"), torch.float32)
    assert offsets.shape == (1000, 2)
    assert torch.all(offsets.abs() <= dr.COM_OFFSET_RANGE_M)


def test_com_offset_disturbance_torque_zero_when_offset_is_zero():
    offset = torch.zeros((1, 2))
    force = torch.tensor([[0.0, 0.0, 5.0]])
    torque = dr.com_offset_disturbance_torque(offset, force)
    assert torch.allclose(torque, torch.zeros_like(torque))


def test_com_offset_disturbance_torque_matches_cross_product_sign_convention():
    # dx=0, dy=+0.01 with Fz=5.0 -> tau_x = dy*Fz > 0, tau_y = -dx*Fz = 0.
    # Matches the same right-hand-rule sign convention as
    # physics.mixer_forces_and_torques (tau_x = +y*Fz, tau_y = -x*Fz).
    offset = torch.tensor([[0.0, 0.01]])
    force = torch.tensor([[0.0, 0.0, 5.0]])
    torque = dr.com_offset_disturbance_torque(offset, force)
    assert torque[0, 0].item() == pytest.approx(0.05, rel=1e-4)
    assert torque[0, 1].item() == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Motor efficiency
# --------------------------------------------------------------------------


def test_sample_motor_efficiency_within_range_and_per_motor():
    eff = dr.sample_motor_efficiency(500, torch.device("cpu"), torch.float32)
    assert eff.shape == (500, 4)
    low, high = dr.MOTOR_EFFICIENCY_RANGE
    assert torch.all(eff >= low)
    assert torch.all(eff <= high)
    # Not all 4 columns identical across envs (independent per motor) --
    # extremely unlikely to fail by chance with 500 samples if truly random.
    assert not torch.allclose(eff[:, 0], eff[:, 1])


def test_apply_motor_efficiency_scales_correctly():
    thrusts = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    eff = torch.tensor([[0.9, 1.0, 1.05, 0.85]])
    out = dr.apply_motor_efficiency(thrusts, eff)
    assert torch.allclose(out, torch.tensor([[0.9, 2.0, 3.15, 3.4]]))


# --------------------------------------------------------------------------
# Action latency
# --------------------------------------------------------------------------


def test_sample_action_delay_steps_within_range():
    delays = dr.sample_action_delay_steps(1000, torch.device("cpu"))
    assert torch.all(delays >= dr.ACTION_DELAY_MIN_STEPS)
    assert torch.all(delays <= dr.ACTION_DELAY_MAX_STEPS)


def test_push_and_read_delayed_action_round_trip():
    num_envs = 2
    history = torch.zeros((num_envs, dr.ACTION_HISTORY_LEN, 4))
    actions_over_time = [
        torch.full((num_envs, 4), 1.0),
        torch.full((num_envs, 4), 2.0),
        torch.full((num_envs, 4), 3.0),
        torch.full((num_envs, 4), 4.0),
    ]
    for a in actions_over_time:
        history = dr.push_action_history(history, a)

    # After pushing 1,2,3,4 in order, history[:,0]=4 (most recent), [:,1]=3,
    # [:,2]=2, [:,3]=1 (oldest still retained, ACTION_HISTORY_LEN=4).
    delay0 = dr.read_delayed_action(history, torch.tensor([0, 0]))
    delay1 = dr.read_delayed_action(history, torch.tensor([1, 1]))
    delay3 = dr.read_delayed_action(history, torch.tensor([3, 3]))
    assert torch.allclose(delay0, torch.full((num_envs, 4), 4.0))
    assert torch.allclose(delay1, torch.full((num_envs, 4), 3.0))
    assert torch.allclose(delay3, torch.full((num_envs, 4), 1.0))


def test_read_delayed_action_per_env_independent_delay():
    history = torch.zeros((2, dr.ACTION_HISTORY_LEN, 4))
    for a in [torch.tensor([[1.0] * 4, [10.0] * 4]),
              torch.tensor([[2.0] * 4, [20.0] * 4]),
              torch.tensor([[3.0] * 4, [30.0] * 4])]:
        history = dr.push_action_history(history, a)
    # env 0 wants delay=2 (-> value 1.0), env 1 wants delay=0 (-> value 30.0)
    out = dr.read_delayed_action(history, torch.tensor([2, 0]))
    assert torch.allclose(out[0], torch.full((4,), 1.0))
    assert torch.allclose(out[1], torch.full((4,), 30.0))


# --------------------------------------------------------------------------
# Observation noise
# --------------------------------------------------------------------------


def test_add_gaussian_noise_zero_std_is_noop():
    value = torch.tensor([1.0, 2.0, 3.0])
    out = dr.add_gaussian_noise(value, 0.0)
    assert torch.equal(out, value)


def test_add_gaussian_noise_nonzero_std_changes_value_and_matches_scale_order():
    value = torch.zeros(10000)
    out = dr.add_gaussian_noise(value, 0.05)
    assert not torch.equal(out, value)
    assert out.std().item() == pytest.approx(0.05, rel=0.15)


def test_perturb_quat_zero_std_is_noop():
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = dr.perturb_quat(quat, 0.0)
    assert torch.equal(out, quat)


def test_perturb_quat_nonzero_std_stays_unit_norm_and_close_to_identity():
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = dr.perturb_quat(quat, 0.002)
    assert out.shape == (1, 4)
    norm = torch.linalg.norm(out, dim=-1)
    assert norm.item() == pytest.approx(1.0, abs=1e-5)
    # Small noise -> should stay very close to the identity quaternion.
    assert torch.allclose(out, quat, atol=0.02)


# --------------------------------------------------------------------------
# Wind gusts
# --------------------------------------------------------------------------


def test_wind_gust_state_starts_with_no_active_gust():
    state = dr.WindGustState.create(4, torch.device("cpu"), torch.float32)
    force, torque = dr.update_wind_gusts(state, control_dt_s=0.02)
    # With a fresh state (no gust yet triggered this call unless the RNG
    # happened to roll a trigger) force/torque should be finite and shaped
    # correctly regardless.
    assert force.shape == (4, 3)
    assert torque.shape == (4, 3)


def test_wind_gust_force_magnitude_bounded_when_forced_active():
    state = dr.WindGustState.create(4, torch.device("cpu"), torch.float32)
    # Force every env into an already-active gust with known bounds, then
    # confirm update_wind_gusts respects and decrements it correctly.
    state.remaining_steps[:] = 5
    state.force_b[:] = torch.tensor([dr.WIND_GUST_FORCE_MAX_N, 0.0, 0.0])
    state.torque_b[:] = torch.tensor([dr.WIND_GUST_TORQUE_MAX_NM, 0.0, 0.0])
    force, torque = dr.update_wind_gusts(state, control_dt_s=0.02)
    assert torch.allclose(force, state.force_b.new_tensor([[dr.WIND_GUST_FORCE_MAX_N, 0.0, 0.0]] * 4))
    assert torch.all(state.remaining_steps == 4)


def test_wind_gust_reset_clears_state_for_selected_envs():
    state = dr.WindGustState.create(4, torch.device("cpu"), torch.float32)
    state.remaining_steps[:] = 5
    state.force_b[:] = 1.0
    state.torque_b[:] = 1.0
    state.reset(torch.tensor([0, 2]))
    assert state.remaining_steps[0].item() == 0
    assert state.remaining_steps[2].item() == 0
    assert state.remaining_steps[1].item() == 5
    assert torch.all(state.force_b[0] == 0.0)
    assert torch.all(state.force_b[2] == 0.0)
    assert torch.all(state.force_b[1] == 1.0)


def test_wind_gust_eventually_triggers_over_many_steps():
    # With a nonzero trigger probability, across enough steps at least one
    # of many envs should end up with an active gust.
    state = dr.WindGustState.create(64, torch.device("cpu"), torch.float32)
    saw_active_gust = False
    for _ in range(200):
        force, torque = dr.update_wind_gusts(state, control_dt_s=0.02)
        if torch.any(force.abs().sum(dim=-1) > 0.0):
            saw_active_gust = True
            break
    assert saw_active_gust
