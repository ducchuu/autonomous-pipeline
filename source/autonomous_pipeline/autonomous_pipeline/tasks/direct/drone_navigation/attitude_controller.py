"""Simulated flight-controller inner loop: converts RC-style commands
(roll rate, pitch rate, yaw rate, throttle -- the same action space the real
drone will receive over ExpressLRS from the RadioMaster TX16S) into a
desired body-frame torque + collective thrust wrench.

WHY THIS EXISTS (sim-to-real architecture, confirmed with the user
2026-08-18): the trained policy will NOT command individual motors. It runs
on the ground PC, reading OptiTrack pose+velocity, and outputs 4 values in
[-1, 1] that get translated into standard RC channel values (Roll, Pitch,
Yaw, Throttle) sent over ExpressLRS to the real flight controller. The FC's
own onboard PID loop (running on ITS OWN IMU, not visible to the policy)
turns those channel commands into motor outputs. For the simulation to
train a policy that transfers to this real pipeline, Isaac Lab must present
the SAME action interface -- so this module stands in for "the real FC's
inner loop" inside the simulator.

============================================================================
ARCHITECTURE HISTORY -- ANGLE mode -> full ACRO/rate mode (2026-08-19)
============================================================================
This module originally ran roll/pitch in Betaflight ANGLE (self-leveling)
mode and only yaw in rate mode. That assumption was invalidated by reading
the user's real `diff all` CLI export from their Betaflight 2025.12.2 /
TMPACERF7 flight controller: it configures exactly ONE `aux` range (the ARM
switch), with no ANGLE/HORIZON aux range at all -- meaning the real FC
defaults to ACRO (full rate mode, no self-leveling) on every axis. The user
was asked to choose between (a) adding a real ANGLE aux switch to their FC
to match the old sim, or (b) redesigning the sim to run ACRO on all three
axes. They explicitly chose (b): "A reinforcement learning policy has a
much higher performance ceiling and tighter control loop when operating in
raw Acro (Rate) mode, as it removes the flight controller's self-leveling
interference."

So, as of 2026-08-19, ALL THREE axes (roll, pitch, yaw) run in rate mode:
the policy's [-1, 1] output maps directly to a target angular velocity
(never a target angle), and this module's job is purely a rate-tracking
inner loop -- exactly what a real ACRO-mode FC's inner loop does. Because
of this, the controller here no longer needs the drone's absolute
orientation at all (a real ACRO FC's rate PID loop only ever reads the
gyro) -- `compute_desired_wrench` takes body angular velocity, not a
quaternion. `quat_to_roll_pitch` is kept as a standalone, independently
tested utility (e.g. for logging/diagnostics or future reward shaping) but
is no longer called from this module's control path.

============================================================================
GAIN DERIVATION -- why this is NOT a literal 1:1 Betaflight PID swap
============================================================================
The user provided their real Betaflight PID/FF profile (`diff all`,
2026-08-19, see DRONE_SPEC.md section 9) and asked to "swap in" these exact
numbers for the inner-loop PID math:

    Roll:  P=45  I=80  D=30  FF=120
    Pitch: P=47  I=84  D=34  FF=125
    Yaw:   P=45  I=80  D=0   FF=120

These numbers CANNOT be substituted literally as N*m / (rad/s) gains --
Betaflight's PID tool values are firmware-internal, unitless quantities
scaled by that firmware's own internal fixed-point math, motor-mixer
normalization, and gyro/looptime scaling. They are not directly comparable
across axes either: Betaflight can use similar-looking P values for roll,
pitch, AND yaw (45/47/45 here) because its own mixer already accounts for
the fact that yaw's actuation channel (small reactive motor torque) is
physically much weaker than roll/pitch's (large differential-thrust torque)
-- copying "P=45" verbatim onto this sim's yaw axis would size yaw's gain
using roll's physical torque budget, which is ~4x too large for what yaw
can actually deliver (see MAX_REACTIVE_TORQUE_NM vs
MAX_DIFFERENTIAL_TORQUE_NM below) and would either saturate constantly or
require Betaflight's internal normalization we don't have.

Instead, this module uses a principled two-step derivation that keeps
faith with the user's real tuning data without pretending the units match:

  1. ABSOLUTE gain magnitude (KP_ROLL, KP_PITCH, KP_YAW) is derived
     per-axis from THIS DRONE'S OWN real torque budget (computed from the
     mixer geometry in physics.py, exactly the same way KP_ANGLE=3.0 was
     originally reasoned) -- so a full-scale rate error never demands more
     than TORQUE_BUDGET_FRACTION of that axis's actual available torque.
     This is what makes the gain physically meaningful in N*m/(rad/s).
  2. RELATIVE shape (I/P, D/P, FF/P ratios) is taken directly from the
     user's real Betaflight profile, per axis. This preserves the real,
     meaningful tuning information in that profile -- e.g. yaw's D=0
     (Betaflight yaw D is conventionally near-zero; noise-prone reactive-
     torque axis), pitch running slightly "hotter" than roll (higher
     P/I/D, matching this drone's slightly different pitch inertia), and
     feedforward weighted far above P on every axis (FF/P ~2.6 on all
     three) -- while letting each axis's ABSOLUTE scale come from step 1.

Net effect: KI_axis = KP_axis * (BF_I_axis / BF_P_axis), and likewise for
KD_axis and KFF_axis. Tagged [DERIVED-FROM-BETAFLIGHT-RATIOS,
ESTIMATE-VERIFY] throughout -- correct in spirit and structurally faithful
to the real tuning, but not a rigorous system-identified match. If you
later run a real step-response test on the actual drone (see rule 9 in
`.agents/rules/isaac_lab_drone_project.md`), replace this derivation with
directly fitted gains.

============================================================================
POLICY AUTHORITY LIMITS (safety-critical -- see DRONE_SPEC.md section 13
"Policy authority limits")
============================================================================
MAX_BODY_RATE_RAD_S, IDLE_THROTTLE_FRACTION, and MAX_THROTTLE_FRACTION below
are NOT modeling Betaflight's own rate/expo curve -- they are the
deliberate, hard authority ceiling the RL policy is allowed to command at
all times (never let the policy request a rate/thrust the FC would
obediently but unsafely execute).

MAX_BODY_RATE_RAD_S = 300 deg/s, set explicitly by the user on 2026-08-19,
intentionally well below Betaflight's real configured "Max Rate" of
BETAFLIGHT_MAX_RATE_DEG_S = 670 deg/s ("Do not give the policy access to
the full 670 deg/s yet; 300 deg/s is plenty fast for [training in the
OptiTrack cage] and prevents erratic spinning during early training"). This
is a strict SUBSET of the real FC's rate range, not a redefinition of it --
raising this toward 670 deg/s later is a legitimate future step once the
policy is well trained, not a bug fix.

IMPORTANT -- yaw authority change, must be flagged and mirrored: the
previous yaw rate ceiling was 60 deg/s. It is now 300 deg/s (5x larger),
because the user asked for ONE shared +/-300 deg/s cap across roll, pitch,
AND yaw. Per rule 7 in `.agents/rules/isaac_lab_drone_project.md`, ANY
authority-limit change must be mirrored EXACTLY in the real
RadioMaster/ExpressLRS inference-time script -- if that script still clamps
yaw to +/-60 deg/s while this sim now trains a policy that expects up to
+/-300 deg/s of yaw authority, sim-to-real transfer fails outright (the
policy will have learned to use yaw-rate dynamics the real inference
script won't let it execute). Copy MAX_BODY_RATE_RAD_S (or the equivalent
RC channel scaling formula) and the throttle constants verbatim into that
script; do not re-derive them independently.

No isaaclab/isaacsim dependency -- pure torch, unit-testable in isolation
(see tests/test_attitude_controller.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import physics

# --------------------------------------------------------------------------
# Policy authority limits -- SAFETY-CRITICAL, see module docstring above.
# --------------------------------------------------------------------------
BETAFLIGHT_MAX_RATE_DEG_S = 670.0  # [MEASURED] real FC "Max Rate" (all axes), `diff all` 2026-08-19
MAX_BODY_RATE_RAD_S = math.radians(300.0)  # [SET BY USER 2026-08-19] shared roll/pitch/yaw ceiling

# Throttle mapping: -1.0 -> a safe idle floor (motors keep spinning, never a
# hard cut mid-air), +1.0 -> capped well below 100% so the policy can never
# command a ceiling-strike climb rate. Unchanged by the ACRO redesign.
# [ESTIMATE-VERIFY: tune within the stated safe band.]
IDLE_THROTTLE_FRACTION = 0.05  # [ESTIMATE-VERIFY] fraction of MAX_TOTAL_THRUST_N at stick -1.0
MAX_THROTTLE_FRACTION = 0.65   # [ESTIMATE-VERIFY] safe band: 60-70% of MAX_TOTAL_THRUST_N at stick +1.0

# --------------------------------------------------------------------------
# Real Betaflight PID/FF profile (`diff all`, 2026-08-19; see DRONE_SPEC.md
# section 9). [MEASURED] -- these are the user's actual firmware values,
# used ONLY for their per-axis RATIOS (I/P, D/P, FF/P) -- see the long
# "GAIN DERIVATION" comment in the module docstring for why.
# --------------------------------------------------------------------------
_AXES = ("roll", "pitch", "yaw")
_BF_P = {"roll": 45.0, "pitch": 47.0, "yaw": 45.0}
_BF_I = {"roll": 80.0, "pitch": 84.0, "yaw": 80.0}
_BF_D = {"roll": 30.0, "pitch": 34.0, "yaw": 0.0}
_BF_FF = {"roll": 120.0, "pitch": 125.0, "yaw": 120.0}

# --------------------------------------------------------------------------
# Per-axis torque budget, derived from this drone's real mixer geometry
# (physics.py) -- NOT a guess. Roll and pitch share the same budget by the
# frame's 4-fold symmetry (differential thrust across two motor pairs);
# yaw's budget is much smaller because it only has each rotor's reactive
# aerodynamic torque to work with, not differential thrust. This asymmetry
# is exactly why Betaflight's own P=45 for BOTH roll and yaw cannot be
# copied verbatim -- see module docstring.
# --------------------------------------------------------------------------
_layout = physics.motor_layout()
_MAX_ARM_Y_M = max(abs(y) for _, y in _layout.positions_xy_m)
MAX_DIFFERENTIAL_TORQUE_NM = 2.0 * _MAX_ARM_Y_M * physics.MAX_THRUST_PER_MOTOR_N  # [DERIVED] roll & pitch
MAX_REACTIVE_TORQUE_NM = (
    2.0 * physics.TORQUE_TO_THRUST_RATIO_M * physics.MAX_THRUST_PER_MOTOR_N
)  # [DERIVED] yaw

# Fraction of an axis's max torque that a FULL-SCALE rate error (at
# MAX_BODY_RATE_RAD_S) is allowed to demand from the P term alone --
# leaves headroom for I/D/FF and disturbance rejection without saturating
# every aggressive command. Same "roughly half the budget" philosophy as
# the original single-stage KP_ANGLE=3.0 design.
TORQUE_BUDGET_FRACTION = 0.5  # [ESTIMATE-VERIFY]


def _kp_from_budget(max_torque_nm: float) -> float:
    return TORQUE_BUDGET_FRACTION * max_torque_nm / MAX_BODY_RATE_RAD_S


def _ratio(axis: str, table: dict[str, float]) -> float:
    """Betaflight's (term / P) ratio for this axis -- dimensionless,
    describes the real FC's internal PID *shape*, independent of its
    (non-transferable) absolute units."""
    return table[axis] / _BF_P[axis]


# --- absolute P gain: from THIS drone's own torque budget (step 1 above) ---
KP_ROLL = _kp_from_budget(MAX_DIFFERENTIAL_TORQUE_NM)   # [DERIVED] N*m per (rad/s)
KP_PITCH = KP_ROLL * (_BF_P["pitch"] / _BF_P["roll"])   # [DERIVED-FROM-BETAFLIGHT-RATIOS]
KP_YAW = _kp_from_budget(MAX_REACTIVE_TORQUE_NM)        # [DERIVED] N*m per (rad/s)

# --- I / D / FF: Betaflight's real per-axis ratios applied to the above ---
KI_ROLL = KP_ROLL * _ratio("roll", _BF_I)      # [DERIVED-FROM-BETAFLIGHT-RATIOS]
KD_ROLL = KP_ROLL * _ratio("roll", _BF_D)      # [DERIVED-FROM-BETAFLIGHT-RATIOS]
KFF_ROLL = KP_ROLL * _ratio("roll", _BF_FF)    # [DERIVED-FROM-BETAFLIGHT-RATIOS]

KI_PITCH = KP_PITCH * _ratio("pitch", _BF_I)   # [DERIVED-FROM-BETAFLIGHT-RATIOS]
KD_PITCH = KP_PITCH * _ratio("pitch", _BF_D)   # [DERIVED-FROM-BETAFLIGHT-RATIOS]
KFF_PITCH = KP_PITCH * _ratio("pitch", _BF_FF)  # [DERIVED-FROM-BETAFLIGHT-RATIOS]

KI_YAW = KP_YAW * _ratio("yaw", _BF_I)         # [DERIVED-FROM-BETAFLIGHT-RATIOS]
KD_YAW = KP_YAW * _ratio("yaw", _BF_D)         # == 0.0, matches Betaflight's real yaw D=0
KFF_YAW = KP_YAW * _ratio("yaw", _BF_FF)       # [DERIVED-FROM-BETAFLIGHT-RATIOS]

# Gains ordered (roll, pitch, yaw) to match the (tau_x, tau_y, tau_z) /
# (rate_x, rate_y, rate_z) convention used throughout this module.
_KP = (KP_ROLL, KP_PITCH, KP_YAW)
_KI = (KI_ROLL, KI_PITCH, KI_YAW)
_KD = (KD_ROLL, KD_PITCH, KD_YAW)
_KFF = (KFF_ROLL, KFF_PITCH, KFF_YAW)

# --------------------------------------------------------------------------
# Integral anti-windup: clamp each axis's ACCUMULATED integral error so its
# torque contribution (KI * integral_error) can never exceed
# INTEGRAL_TORQUE_FRACTION of that axis's real torque budget, regardless of
# how long a large rate error persists (e.g. a stuck/saturated command).
# --------------------------------------------------------------------------
INTEGRAL_TORQUE_FRACTION = 0.3  # [ESTIMATE-VERIFY]
_INTEGRAL_LIMIT_ROLL = INTEGRAL_TORQUE_FRACTION * MAX_DIFFERENTIAL_TORQUE_NM / KI_ROLL
_INTEGRAL_LIMIT_PITCH = INTEGRAL_TORQUE_FRACTION * MAX_DIFFERENTIAL_TORQUE_NM / KI_PITCH
_INTEGRAL_LIMIT_YAW = INTEGRAL_TORQUE_FRACTION * MAX_REACTIVE_TORQUE_NM / KI_YAW
_INTEGRAL_LIMIT = (_INTEGRAL_LIMIT_ROLL, _INTEGRAL_LIMIT_PITCH, _INTEGRAL_LIMIT_YAW)

# Minimum dt used in any division below -- guards against a pathological
# dt_s=0 call producing inf/nan in the D and FF terms.
_MIN_DT_S = 1.0e-6


def _axis_tensor(values: tuple[float, float, float], reference: torch.Tensor) -> torch.Tensor:
    """Build a (3,) tensor from a (roll, pitch, yaw) constant tuple, on the
    same device/dtype as `reference`, ready to broadcast against a
    (num_envs, 3) tensor."""
    return torch.tensor(values, device=reference.device, dtype=reference.dtype)


@dataclass
class RateControllerState:
    """Persistent per-env state for the rate-mode P-I-D-FF inner loop.

    A real FC's rate PID loop is stateful (integral accumulator, D-on-gyro,
    feedforward-on-setpoint-change) -- unlike the old single-stage
    stateless angle-mode PD controller, `compute_desired_wrench` now needs
    this state carried across control steps. Mirrors the
    `domain_randomization.WindGustState` create()/reset() pattern used
    elsewhere in this codebase.

    integral_error:   (num_envs, 3) rad -- accumulated (roll, pitch, yaw)
                       rate error, integrated over time, anti-windup
                       clamped every step (see _INTEGRAL_LIMIT above).
    prev_rate_actual: (num_envs, 3) rad/s -- previous step's ACTUAL gyro
                       rate, for the D term ("D on gyro" / D-on-measurement,
                       matching Betaflight's real behavior and avoiding
                       derivative kick from a setpoint step).
    prev_rate_target: (num_envs, 3) rad/s -- previous step's TARGET rate,
                       for the feedforward term (reacts to setpoint
                       CHANGES only, exactly like a real FC's FF term).
    """

    integral_error: torch.Tensor
    prev_rate_actual: torch.Tensor
    prev_rate_target: torch.Tensor

    @classmethod
    def create(cls, num_envs: int, device: torch.device, dtype: torch.dtype) -> "RateControllerState":
        return cls(
            integral_error=torch.zeros((num_envs, 3), device=device, dtype=dtype),
            prev_rate_actual=torch.zeros((num_envs, 3), device=device, dtype=dtype),
            prev_rate_target=torch.zeros((num_envs, 3), device=device, dtype=dtype),
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self.integral_error[env_ids] = 0.0
        self.prev_rate_actual[env_ids] = 0.0
        self.prev_rate_target[env_ids] = 0.0


def quat_to_roll_pitch(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract roll and pitch (radians) from a (num_envs, 4) quaternion in
    (w, x, y, z) order, using the standard aerospace ZYX Euler extraction.

    NOTE (2026-08-19): no longer called from `compute_desired_wrench` --
    the controller is now full ACRO/rate mode on all axes and, like a real
    ACRO-mode FC, only needs gyro (angular velocity), never an absolute
    attitude estimate. Kept as a standalone, independently tested utility
    for diagnostics/logging or future reward shaping that may still want a
    roll/pitch angle reading.
    """
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation), clamp for the gimbal-lock edge case
    sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)

    return roll, pitch


def compute_desired_wrench(
    actions: torch.Tensor,
    current_ang_vel_b: torch.Tensor,
    controller_state: RateControllerState,
    dt_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulated FC inner loop: RC-style action -> desired (force_z, torque).

    Full ACRO/rate mode on roll, pitch, AND yaw (see module docstring
    "ARCHITECTURE HISTORY" for why) -- every axis maps the policy's [-1, 1]
    output directly to a target body rate (never a target angle), and this
    function runs one P-I-D-FF step of a rate-tracking inner loop per axis,
    mutating `controller_state` in place (integral accumulator, previous
    actual rate, previous target rate).

    actions:            (num_envs, 4) in [-1, 1] = [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd]
    current_ang_vel_b:  (num_envs, 3) body-frame angular velocity (roll, pitch,
                         yaw rate), rad/s -- i.e. the gyro reading a real ACRO
                         FC's rate loop runs on.
    controller_state:   RateControllerState, persistent across control steps;
                         call `.reset(env_ids)` on episode reset for the
                         affected envs (see drone_navigation_env.py).
    dt_s:                control-step duration in seconds (this env runs the
                         inner loop once per control step, i.e. every
                         `decimation` physics steps -- see
                         DroneTaskEnv._control_dt_s).

    returns:
        desired_force_z: (num_envs,) newtons, collective thrust demand
        desired_torque:  (num_envs, 3) N*m, (tau_x, tau_y, tau_z) body torque demand

    Feed the outputs into physics.inverse_mixer(...) to get the 4 per-motor
    thrusts that best achieve this wrench within motor saturation limits.
    """
    dt_s = max(dt_s, _MIN_DT_S)
    actions = actions.clamp(-1.0, 1.0)
    rate_cmd = actions[..., 0:3]  # (num_envs, 3) = (roll_cmd, pitch_cmd, yaw_cmd)
    throttle_cmd = actions[..., 3]

    rate_target = rate_cmd * MAX_BODY_RATE_RAD_S  # (num_envs, 3), rad/s
    rate_actual = current_ang_vel_b  # (num_envs, 3), rad/s -- already (roll, pitch, yaw)
    rate_error = rate_target - rate_actual

    kp = _axis_tensor(_KP, rate_error)
    ki = _axis_tensor(_KI, rate_error)
    kd = _axis_tensor(_KD, rate_error)
    kff = _axis_tensor(_KFF, rate_error)
    integral_limit = _axis_tensor(_INTEGRAL_LIMIT, rate_error)

    # --- integral term (anti-windup clamped) ---
    new_integral = controller_state.integral_error + rate_error * dt_s
    new_integral = torch.clamp(new_integral, min=-integral_limit, max=integral_limit)

    # --- D term, "D on gyro" (on measurement, not on error) -- avoids
    # derivative kick from a setpoint step, matches Betaflight's real
    # behavior. Sign: opposes the rate of change of the ACTUAL rate. ---
    d_term = -kd * (rate_actual - controller_state.prev_rate_actual) / dt_s

    # --- feedforward term -- reacts to setpoint CHANGES only, matching a
    # real FC's FF term (predictive push that cancels P/I lag when the
    # target rate itself is moving, e.g. right after the policy changes
    # its commanded stick position). ---
    ff_term = kff * (rate_target - controller_state.prev_rate_target) / dt_s

    desired_torque = kp * rate_error + ki * new_integral + d_term + ff_term

    controller_state.integral_error = new_integral
    controller_state.prev_rate_actual = rate_actual.clone()
    controller_state.prev_rate_target = rate_target.clone()

    # Throttle mapping with a hard authority ceiling AND an idle floor (see
    # "Policy authority limits" above) -- NOT a bare [-1,1] -> [0,1] map to
    # [0, MAX_TOTAL_THRUST_N]. -1.0 always requests IDLE_THROTTLE_FRACTION of
    # max thrust (never zero, mirrors a real ESC idle so motors never fully
    # stop mid-air), +1.0 is capped at MAX_THROTTLE_FRACTION so the policy
    # can never command a climb rate that would strike the 2.5m ceiling.
    # Unchanged by the ACRO redesign.
    throttle_norm = (throttle_cmd + 1.0) * 0.5  # [-1,1] -> [0,1]
    throttle_scaled = IDLE_THROTTLE_FRACTION + throttle_norm * (
        MAX_THROTTLE_FRACTION - IDLE_THROTTLE_FRACTION
    )
    desired_force_z = throttle_scaled * physics.MAX_TOTAL_THRUST_N

    return desired_force_z, desired_torque
