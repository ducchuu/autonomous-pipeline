"""Pure-Python unit tests for attitude_controller.py + physics.inverse_mixer.

Run with: pytest tests/test_attitude_controller.py -v

Covers the full ACRO/rate-mode redesign (2026-08-19): all three axes
(roll, pitch, yaw) now run a stateful rate-tracking P-I-D-FF inner loop,
replacing the old stateless angle-mode PD controller for roll/pitch.
"""

import importlib.util
import math
import sys
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


physics = _load("drone_physics_for_attitude_test", "physics.py")

# attitude_controller.py does `from . import physics` (relative import), which
# requires a real package context. Since we only need it for standalone
# testing (no isaaclab), fake a minimal parent package pointing back at this
# same physics module before loading it by path.
import types  # noqa: E402

_fake_pkg = types.ModuleType("drone_navigation_pkg_for_test")
_fake_pkg.physics = physics
sys.modules["drone_navigation_pkg_for_test"] = _fake_pkg
sys.modules["drone_navigation_pkg_for_test.physics"] = physics

_spec = importlib.util.spec_from_file_location(
    "drone_navigation_pkg_for_test.attitude_controller",
    _TASK_DIR / "attitude_controller.py",
)
attitude_controller = importlib.util.module_from_spec(_spec)
attitude_controller.__package__ = "drone_navigation_pkg_for_test"
sys.modules[_spec.name] = attitude_controller
_spec.loader.exec_module(attitude_controller)


IDENTITY_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
ZERO_ANG_VEL = torch.zeros((1, 3))
DT_S = 0.02  # matches the real 50Hz control loop (sim.dt * decimation)


def _fresh_state(num_envs: int = 1) -> "attitude_controller.RateControllerState":
    return attitude_controller.RateControllerState.create(num_envs, torch.device("cpu"), torch.float32)


def test_quat_to_roll_pitch_identity_is_zero():
    # Standalone utility, no longer on the control path (see module
    # docstring "ARCHITECTURE HISTORY") but still independently tested.
    roll, pitch = attitude_controller.quat_to_roll_pitch(IDENTITY_QUAT)
    assert roll.item() == pytest.approx(0.0, abs=1e-6)
    assert pitch.item() == pytest.approx(0.0, abs=1e-6)


def test_level_hover_command_produces_zero_torque_on_all_axes():
    # centered sticks (roll=pitch=yaw=0), fresh state, no rotation -> the
    # rate-mode P-I-D-FF controller should command zero torque on every
    # axis (target rate == actual rate == 0, no history to differentiate).
    actions = torch.tensor([[0.0, 0.0, 0.0, -1.0]])
    force_z, torque = attitude_controller.compute_desired_wrench(
        actions, ZERO_ANG_VEL, _fresh_state(), DT_S
    )
    assert torque[0, 0].item() == pytest.approx(0.0, abs=1e-9)
    assert torque[0, 1].item() == pytest.approx(0.0, abs=1e-9)
    assert torque[0, 2].item() == pytest.approx(0.0, abs=1e-9)


def test_full_throttle_is_capped_at_max_throttle_fraction_not_100_percent():
    # Safety-critical: full-stick throttle must NOT reach 100% of the
    # drone's total thrust capacity -- it must be capped at
    # MAX_THROTTLE_FRACTION (60-70% band) so the policy can never command a
    # ceiling-strike climb rate. Unchanged by the ACRO redesign.
    actions = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    force_z, _ = attitude_controller.compute_desired_wrench(
        actions, ZERO_ANG_VEL, _fresh_state(), DT_S
    )
    expected = attitude_controller.MAX_THROTTLE_FRACTION * physics.MAX_TOTAL_THRUST_N
    assert force_z.item() == pytest.approx(expected, rel=1e-4)
    assert force_z.item() < physics.MAX_TOTAL_THRUST_N


def test_zero_stick_throttle_gives_idle_floor_not_zero_thrust():
    # Safety-critical: stick -1.0 must still request a small idle thrust
    # (motors never fully stop mid-air), not exactly zero. Unchanged by the
    # ACRO redesign.
    actions = torch.tensor([[0.0, 0.0, 0.0, -1.0]])
    force_z, _ = attitude_controller.compute_desired_wrench(
        actions, ZERO_ANG_VEL, _fresh_state(), DT_S
    )
    expected = attitude_controller.IDLE_THROTTLE_FRACTION * physics.MAX_TOTAL_THRUST_N
    assert force_z.item() == pytest.approx(expected, rel=1e-4)
    assert force_z.item() > 0.0


def test_body_rate_authority_limit_is_300_deg_s_and_well_below_the_real_fc_max():
    # Safety-critical: guards against accidentally widening the policy's
    # rate authority back toward Betaflight's real 670 deg/s Max Rate. The
    # user explicitly capped the policy at 300 deg/s on all 3 axes even
    # though the real FC allows up to 670 deg/s -- see module docstring
    # "POLICY AUTHORITY LIMITS".
    assert attitude_controller.MAX_BODY_RATE_RAD_S == pytest.approx(math.radians(300.0))
    assert attitude_controller.BETAFLIGHT_MAX_RATE_DEG_S == pytest.approx(670.0)
    assert attitude_controller.MAX_BODY_RATE_RAD_S < math.radians(
        attitude_controller.BETAFLIGHT_MAX_RATE_DEG_S
    )
    assert 0.60 <= attitude_controller.MAX_THROTTLE_FRACTION <= 0.70


def test_yaw_authority_now_matches_roll_pitch_authority():
    # The old design gave yaw only 60 deg/s while roll/pitch (in angle
    # mode) used a completely different limit type. The 2026-08-19 ACRO
    # redesign gives all 3 axes the exact SAME rate ceiling, per the user's
    # explicit instruction. This is a policy-authority-limit CHANGE that
    # must be mirrored in the real RadioMaster inference script (rule 7).
    assert attitude_controller.MAX_BODY_RATE_RAD_S == pytest.approx(math.radians(300.0))


def test_positive_roll_command_produces_positive_roll_torque_when_level():
    actions = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    _, torque = attitude_controller.compute_desired_wrench(
        actions, ZERO_ANG_VEL, _fresh_state(), DT_S
    )
    assert torque[0, 0].item() > 0.0


def test_negative_pitch_command_produces_negative_pitch_torque_when_level():
    actions = torch.tensor([[0.0, -1.0, 0.0, 0.0]])
    _, torque = attitude_controller.compute_desired_wrench(
        actions, ZERO_ANG_VEL, _fresh_state(), DT_S
    )
    assert torque[0, 1].item() < 0.0


def test_yaw_command_produces_yaw_torque():
    actions = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    _, torque = attitude_controller.compute_desired_wrench(
        actions, ZERO_ANG_VEL, _fresh_state(), DT_S
    )
    assert torque[0, 2].item() > 0.0


def test_yaw_axis_has_zero_derivative_gain_matching_betaflight_profile():
    # The user's real Betaflight profile has D=0 on yaw (Roll D=30,
    # Pitch D=34, Yaw D=0) -- this ratio must survive into the derived
    # sim gains exactly, since KD_YAW = KP_YAW * (0 / BF_P_yaw) = 0.
    assert attitude_controller.KD_YAW == pytest.approx(0.0, abs=1e-12)
    assert attitude_controller.KD_ROLL > 0.0
    assert attitude_controller.KD_PITCH > 0.0


def test_pitch_gains_are_slightly_stronger_than_roll_matching_betaflight_ratio():
    # Betaflight P/I/D/FF are all slightly higher for pitch than roll
    # (47/84/34/125 vs 45/80/30/120) -- the derived sim gains should
    # preserve that same relative ordering.
    assert attitude_controller.KP_PITCH > attitude_controller.KP_ROLL
    assert attitude_controller.KI_PITCH > attitude_controller.KI_ROLL
    assert attitude_controller.KD_PITCH > attitude_controller.KD_ROLL
    assert attitude_controller.KFF_PITCH > attitude_controller.KFF_ROLL


def test_yaw_torque_budget_is_much_smaller_than_roll_pitch_reflecting_real_physics():
    # Yaw only has each rotor's small reactive torque to work with, not
    # differential thrust across the frame -- this MUST stay much smaller
    # than roll/pitch's budget, which is exactly why Betaflight's raw
    # P=45 (same order as roll) can't be copied onto yaw's absolute gain.
    assert attitude_controller.MAX_REACTIVE_TORQUE_NM < 0.5 * attitude_controller.MAX_DIFFERENTIAL_TORQUE_NM
    assert attitude_controller.KP_YAW < attitude_controller.KP_ROLL


def test_zero_error_steady_state_after_target_stabilizes_gives_only_pi_torque():
    # Step 1: a stick step from neutral produces a large one-off
    # feedforward transient (target rate jumps from 0 -> commanded value)
    # -- this is EXPECTED (mirrors a real FC's aggressive FF response to a
    # step input) and gets saturated downstream by physics.inverse_mixer.
    # Step 2: with the SAME command held and rate still unmatched, the FF
    # term (which only reacts to setpoint CHANGES) must drop to ~0, since
    # the target didn't move between step 1 and step 2.
    state = _fresh_state()
    actions = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    _, torque_step1 = attitude_controller.compute_desired_wrench(actions, ZERO_ANG_VEL, state, DT_S)
    _, torque_step2 = attitude_controller.compute_desired_wrench(actions, ZERO_ANG_VEL, state, DT_S)
    assert abs(torque_step2[0, 0].item()) < abs(torque_step1[0, 0].item())


def test_integral_accumulates_with_persistent_rate_error():
    # With a constant nonzero rate error held across steps, the integral
    # accumulator must grow monotonically (until anti-windup clamps it).
    state = _fresh_state()
    actions = torch.tensor([[0.2, 0.0, 0.0, 0.0]])
    initial_integral = state.integral_error[0, 0].item()
    for _ in range(3):
        attitude_controller.compute_desired_wrench(actions, ZERO_ANG_VEL, state, DT_S)
    assert state.integral_error[0, 0].item() > initial_integral


def test_integral_is_anti_windup_clamped():
    # Hold a large, persistent rate error for many steps -- the integral
    # accumulator must saturate at the documented anti-windup limit, not
    # grow without bound.
    state = _fresh_state()
    actions = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    for _ in range(2000):
        attitude_controller.compute_desired_wrench(actions, ZERO_ANG_VEL, state, DT_S)
    limit = attitude_controller._INTEGRAL_LIMIT[0]
    assert state.integral_error[0, 0].item() == pytest.approx(limit, rel=1e-3)


def test_reset_zeroes_integral_and_previous_rate_state():
    state = _fresh_state()
    actions = torch.tensor([[1.0, 0.5, -0.3, 0.0]])
    for _ in range(5):
        attitude_controller.compute_desired_wrench(actions, ZERO_ANG_VEL, state, DT_S)
    assert state.integral_error.abs().sum().item() > 0.0

    state.reset(torch.tensor([0]))
    assert torch.all(state.integral_error == 0.0)
    assert torch.all(state.prev_rate_actual == 0.0)
    assert torch.all(state.prev_rate_target == 0.0)


def test_reset_only_affects_the_given_env_ids_in_a_multi_env_batch():
    state = _fresh_state(num_envs=2)
    actions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    ang_vel = torch.zeros((2, 3))
    for _ in range(5):
        attitude_controller.compute_desired_wrench(actions, ang_vel, state, DT_S)
    assert state.integral_error[0, 0].item() != 0.0
    assert state.integral_error[1, 0].item() != 0.0

    state.reset(torch.tensor([0]))
    assert state.integral_error[0, 0].item() == 0.0
    assert state.integral_error[1, 0].item() != 0.0


def test_actions_are_clamped_to_valid_range_before_use():
    # Out-of-range actions (e.g. from an untrained/exploring policy before
    # a tanh squash is applied) must be clamped, not silently overshoot the
    # authority ceiling.
    actions_over = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    actions_at_max = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    _, torque_over = attitude_controller.compute_desired_wrench(actions_over, ZERO_ANG_VEL, _fresh_state(), DT_S)
    _, torque_at_max = attitude_controller.compute_desired_wrench(actions_at_max, ZERO_ANG_VEL, _fresh_state(), DT_S)
    assert torque_over[0, 0].item() == pytest.approx(torque_at_max[0, 0].item(), rel=1e-6)


def test_inverse_mixer_round_trip_within_saturation():
    desired_force_z = torch.tensor([physics.MASS_TOTAL_KG * physics.GRAVITY_MPS2])
    desired_torque = torch.tensor([[0.05, -0.03, 0.01]])
    motor_thrusts = physics.inverse_mixer(desired_force_z, desired_torque)

    assert motor_thrusts.shape == (1, 4)
    assert torch.all(motor_thrusts >= 0.0)
    assert torch.all(motor_thrusts <= physics.MAX_THRUST_PER_MOTOR_N + 1e-6)

    achieved_force, achieved_torque = physics.mixer_forces_and_torques(motor_thrusts)
    assert achieved_force[0, 2].item() == pytest.approx(desired_force_z.item(), rel=1e-3)
    assert torch.allclose(achieved_torque[0], desired_torque[0], atol=1e-3)


def test_inverse_mixer_clips_when_saturated():
    # Ask for far more torque than the motors can actually deliver.
    desired_force_z = torch.tensor([physics.MAX_TOTAL_THRUST_N])
    desired_torque = torch.tensor([[100.0, 0.0, 0.0]])
    motor_thrusts = physics.inverse_mixer(desired_force_z, desired_torque)
    assert torch.all(motor_thrusts >= 0.0)
    assert torch.all(motor_thrusts <= physics.MAX_THRUST_PER_MOTOR_N + 1e-6)
