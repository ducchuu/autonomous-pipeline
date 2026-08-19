"""Physical model for the 5-inch quadcopter: mass/inertia composition, motor mixer,
and induced-power model.

This module has NO dependency on isaaclab / isaacsim / torch-cuda specifics on
purpose: every function here is plain Python + torch tensor math, so it can be
unit tested with plain `pytest` (see tests/test_physics.py) without ever
booting the simulator.

All numeric sources are documented in ``DRONE_SPEC.md`` at the repository
root. Every constant below is tagged the same way as in that file:

    # [MEASURED]        -> value you gave me directly
    # [DERIVED]         -> computed from measured values
    # [ESTIMATE-VERIFY] -> placeholder pending your own measurement/CAD data

Do not change the *shape* of these functions casually -- drone_navigation_env.py
depends on the exact return signatures. Change the *constants* freely as you
gather better data; nothing else needs to change when you do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# --------------------------------------------------------------------------
# 1. Named physical constants (see DRONE_SPEC.md sections 1-3)
# --------------------------------------------------------------------------

GRAVITY_MPS2 = 9.81  # [standard constant]

# --- masses ---
MASS_TOTAL_KG = 0.54563          # [MEASURED] all-up mass incl. battery
MASS_BATTERY_KG = 0.16663        # [MEASURED]
MASS_MOTOR_KG = 0.0315           # [MEASURED] each of 4 motors
N_MOTORS = 4
MASS_MOTORS_TOTAL_KG = MASS_MOTOR_KG * N_MOTORS                       # [DERIVED]
MASS_CORE_KG = MASS_TOTAL_KG - MASS_BATTERY_KG - MASS_MOTORS_TOTAL_KG  # [DERIVED]

# --- geometry ---
ARM_RADIUS_M = 0.115             # [MEASURED] drone center -> motor center
FRAME_HEIGHT_M = 0.036           # [MEASURED] frame stack bottom -> top plate
CORE_FOOTPRINT_W_M = 0.065       # [ESTIMATE-VERIFY] core (frame+electronics) width
CORE_FOOTPRINT_D_M = 0.065       # [ESTIMATE-VERIFY] core (frame+electronics) depth
BATTERY_HALF_THICKNESS_M = 0.015  # [ESTIMATE-VERIFY] half-thickness of a 4S 1550mAh 100C pack

# Component z-positions relative to a frame-bottom origin (z=0).
Z_CORE_M = FRAME_HEIGHT_M / 2.0                       # [ESTIMATE-VERIFY] core mid-stack
Z_MOTOR_M = FRAME_HEIGHT_M / 2.0                      # [ESTIMATE-VERIFY] arms ~mid-stack
Z_BATTERY_M = FRAME_HEIGHT_M + BATTERY_HALF_THICKNESS_M  # [DERIVED from above]

# --- propulsion ---
MOTOR_KV = 2400                                       # [MEASURED]
MAX_THRUST_PER_MOTOR_GF_AT_4S = 1510.0                # [MEASURED] box rating @ 4S
MAX_THRUST_PER_MOTOR_N = MAX_THRUST_PER_MOTOR_GF_AT_4S / 1000.0 * GRAVITY_MPS2  # [DERIVED]
MAX_TOTAL_THRUST_N = MAX_THRUST_PER_MOTOR_N * 4  # [DERIVED] collective thrust ceiling
TORQUE_TO_THRUST_RATIO_M = 0.020                      # [ESTIMATE-VERIFY] kappa, see DRONE_SPEC.md
PROP_DIAMETER_IN = 5.0                                # [MEASURED class: "5-inch" drone]
PROP_DIAMETER_M = PROP_DIAMETER_IN * 0.0254
PROP_DISK_AREA_M2 = math.pi * (PROP_DIAMETER_M / 2.0) ** 2
AIR_DENSITY_KGPM3 = 1.225                             # [standard sea-level constant]
ROTOR_FIGURE_OF_MERIT = 0.65                          # [ESTIMATE-VERIFY] ideal/actual power ratio

# Ideal (momentum-theory) induced-power coefficient: P_ideal = C_POWER_IDEAL * T^1.5
# Source: https://en.wikipedia.org/wiki/Momentum_theory  (P = sqrt(T^3 / (2 rho A)))
C_POWER_IDEAL = 1.0 / math.sqrt(2.0 * AIR_DENSITY_KGPM3 * PROP_DISK_AREA_M2)  # [DERIVED]
C_POWER_ELECTRICAL = C_POWER_IDEAL / ROTOR_FIGURE_OF_MERIT                    # [DERIVED]

# Standard X-quad motor layout: FR, RR, RL, FL at 45/135/225/315 degrees.
# Spin directions chosen so diagonal pairs share direction (yaw-neutral at
# equal thrust) -- [ESTIMATE-VERIFY] against your actual ESC motor-direction
# configuration.
_MOTOR_ANGLES_DEG = (45.0, 135.0, 225.0, 315.0)
_MOTOR_SPIN_DIRECTIONS = (+1.0, -1.0, +1.0, -1.0)  # +1 = CCW, -1 = CW


@dataclass(frozen=True)
class MassProperties:
    """Result of :func:`compute_mass_properties`."""

    total_mass_kg: float
    com_height_m: float  # above the frame-bottom origin
    inertia_diag_kgm2: tuple[float, float, float]  # (Ixx, Iyy, Izz)


@dataclass(frozen=True)
class MotorLayout:
    """Body-frame motor positions (relative to CoM) and spin directions."""

    positions_xy_m: tuple[tuple[float, float], ...]
    spin_directions: tuple[float, ...]


def motor_layout() -> MotorLayout:
    """Return the 4 motor (x, y) positions in the body frame, in a fixed,
    documented order: [front-right, rear-right, rear-left, front-left].
    """
    positions = tuple(
        (
            round(ARM_RADIUS_M * math.cos(math.radians(angle)), 6),
            round(ARM_RADIUS_M * math.sin(math.radians(angle)), 6),
        )
        for angle in _MOTOR_ANGLES_DEG
    )
    return MotorLayout(positions_xy_m=positions, spin_directions=_MOTOR_SPIN_DIRECTIONS)


def compute_mass_properties() -> MassProperties:
    """Compute total mass, CoM height, and diagonal inertia tensor from the
    discrete component model documented in DRONE_SPEC.md section 2.

    Uses the parallel-axis (Huygens-Steiner) theorem to combine:
      - the "core" block (frame + FC + ESC + canopy + camera + VTX + wiring),
        modeled as a solid cuboid with its own local inertia,
      - 4 motors, modeled as point masses at the arm tips,
      - the battery, modeled as a point mass centered on top of the frame.

    This replaces the single-uniform-cuboid approximation from the original
    guidance, which is not accurate for a drone whose mass is this unevenly
    distributed (see DRONE_SPEC.md section 2 for the full rationale).
    """
    layout = motor_layout()

    # --- center of mass height (mass-weighted average of component z) ---
    numerator = (
        MASS_CORE_KG * Z_CORE_M
        + MASS_MOTORS_TOTAL_KG * Z_MOTOR_M
        + MASS_BATTERY_KG * Z_BATTERY_M
    )
    com_height_m = numerator / MASS_TOTAL_KG

    # --- core block: local cuboid inertia + parallel-axis shift in z only ---
    w, d, h = CORE_FOOTPRINT_W_M, CORE_FOOTPRINT_D_M, FRAME_HEIGHT_M
    ixx_core_local = (1.0 / 12.0) * MASS_CORE_KG * (d**2 + h**2)
    iyy_core_local = (1.0 / 12.0) * MASS_CORE_KG * (w**2 + h**2)
    izz_core_local = (1.0 / 12.0) * MASS_CORE_KG * (w**2 + d**2)
    dz_core = Z_CORE_M - com_height_m
    ixx_core = ixx_core_local + MASS_CORE_KG * dz_core**2
    iyy_core = iyy_core_local + MASS_CORE_KG * dz_core**2
    izz_core = izz_core_local  # centered in x, y -> no shift

    # --- motors: point masses at arm tips ---
    dz_motor = Z_MOTOR_M - com_height_m
    ixx_motors = 0.0
    iyy_motors = 0.0
    izz_motors = 0.0
    for x, y in layout.positions_xy_m:
        ixx_motors += MASS_MOTOR_KG * (y**2 + dz_motor**2)
        iyy_motors += MASS_MOTOR_KG * (x**2 + dz_motor**2)
        izz_motors += MASS_MOTOR_KG * (x**2 + y**2)

    # --- battery: centered point mass above the frame ---
    dz_batt = Z_BATTERY_M - com_height_m
    ixx_batt = MASS_BATTERY_KG * dz_batt**2
    iyy_batt = MASS_BATTERY_KG * dz_batt**2
    izz_batt = 0.0

    ixx = ixx_core + ixx_motors + ixx_batt
    iyy = iyy_core + iyy_motors + iyy_batt
    izz = izz_core + izz_motors + izz_batt

    return MassProperties(
        total_mass_kg=MASS_TOTAL_KG,
        com_height_m=com_height_m,
        inertia_diag_kgm2=(ixx, iyy, izz),
    )


# --------------------------------------------------------------------------
# 2. Vectorized (torch) per-step physics -- used inside DroneTaskEnv
# --------------------------------------------------------------------------


def thrust_from_action(actions: torch.Tensor) -> torch.Tensor:
    # NOTE: not used by DroneTaskEnv's actual control pipeline anymore --
    # see attitude_controller.py + physics.inverse_mixer for the RC-style
    # roll/pitch/yaw/throttle action path that matches the real
    # OptiTrack + ExpressLRS sim-to-real architecture. Kept as a simple,
    # independently testable direct-thrust utility for any future
    # experiment that bypasses the simulated flight-controller inner loop.
    #
    # Map raw per-motor actions in [-1, 1] to thrust in newtons, linearly
    # from 0 to MAX_THRUST_PER_MOTOR_N.
    # actions: (num_envs, 4)
    # returns: (num_envs, 4) thrust in newtons, each in [0, MAX_THRUST_PER_MOTOR_N]
    throttle = (actions.clamp(-1.0, 1.0) + 1.0) * 0.5
    return throttle * MAX_THRUST_PER_MOTOR_N


def mixer_forces_and_torques(thrusts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert 4 per-motor thrusts into a body-frame force vector and torque
    vector applied at the drone's center of mass.

    thrusts: (num_envs, 4) newtons, ordered [FR, RR, RL, FL] matching
             physics.motor_layout().

    returns:
        forces  (num_envs, 3): body-frame force, thrust acts purely along +Z
        torques (num_envs, 3): body-frame (tau_x, tau_y, tau_z)

    Roll/pitch torques come from the thrust moment arm (force x lever-arm);
    yaw torque comes from each rotor's aerodynamic reaction torque, modeled
    as proportional to its thrust via TORQUE_TO_THRUST_RATIO_M (kappa) with
    the spin direction sign -- see DRONE_SPEC.md section 3 and
    https://rpg.ifi.uzh.ch/docs/RAL17_Faessler.pdf for the kappa model.
    """
    layout = motor_layout()
    device = thrusts.device
    dtype = thrusts.dtype
    xs = torch.tensor([p[0] for p in layout.positions_xy_m], device=device, dtype=dtype)
    ys = torch.tensor([p[1] for p in layout.positions_xy_m], device=device, dtype=dtype)
    spins = torch.tensor(layout.spin_directions, device=device, dtype=dtype)

    num_envs = thrusts.shape[0]
    forces = torch.zeros((num_envs, 3), device=device, dtype=dtype)
    forces[:, 2] = thrusts.sum(dim=-1)

    # Roll torque about body X from thrust differential along Y; pitch torque
    # about body Y from thrust differential along X (right-hand rule, Z-up).
    tau_x = torch.sum(thrusts * ys.unsqueeze(0), dim=-1)
    tau_y = -torch.sum(thrusts * xs.unsqueeze(0), dim=-1)
    tau_z = torch.sum(thrusts * spins.unsqueeze(0) * TORQUE_TO_THRUST_RATIO_M, dim=-1)

    torques = torch.stack([tau_x, tau_y, tau_z], dim=-1)
    return forces, torques


def _build_mixer_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build the constant 4x4 matrix M such that [Fz, tau_x, tau_y, tau_z]^T
    = M @ [T1, T2, T3, T4]^T, for this drone's fixed X-quad geometry. Same
    physics as mixer_forces_and_torques, expressed as a matrix so it can be
    inverted once and reused for inverse_mixer().
    """
    layout = motor_layout()
    xs = [p[0] for p in layout.positions_xy_m]
    ys = [p[1] for p in layout.positions_xy_m]
    spins = layout.spin_directions
    rows = []
    rows.append([1.0, 1.0, 1.0, 1.0])  # Fz row
    rows.append(list(ys))  # tau_x row
    rows.append([-x for x in xs])  # tau_y row
    rows.append([s * TORQUE_TO_THRUST_RATIO_M for s in spins])  # tau_z row
    return torch.tensor(rows, device=device, dtype=dtype)


def inverse_mixer(
    desired_force_z: torch.Tensor, desired_torque: torch.Tensor
) -> torch.Tensor:
    """Solve for the 4 per-motor thrusts that best produce a desired
    collective thrust (+Z, newtons) and body torque (tau_x, tau_y, tau_z,
    N*m), then clip each motor to its physical [0, MAX_THRUST_PER_MOTOR_N]
    range.

    This is the exact linear inverse of mixer_forces_and_torques for this
    drone's fixed X-quad geometry: [T1..T4] = M^-1 @ [Fz, tau_x, tau_y, tau_z].
    Used by attitude_controller.py to turn a desired torque/thrust wrench
    (from the simulated flight-controller inner loop) into achievable motor
    commands -- clipping here is what lets saturation show up physically
    (e.g. an aggressive roll command at full throttle may not achieve the
    full commanded torque, exactly like a real quad running out of headroom).

    desired_force_z: (num_envs,) newtons
    desired_torque:  (num_envs, 3) N*m, (tau_x, tau_y, tau_z)
    returns: (num_envs, 4) newtons, each in [0, MAX_THRUST_PER_MOTOR_N]
    """
    device = desired_force_z.device
    dtype = desired_force_z.dtype
    mixer_matrix = _build_mixer_matrix(device, dtype)
    mixer_matrix_inv = torch.linalg.inv(mixer_matrix)

    wrench = torch.cat([desired_force_z.unsqueeze(-1), desired_torque], dim=-1)  # (num_envs, 4)
    motor_thrusts = wrench @ mixer_matrix_inv.T
    return motor_thrusts.clamp(min=0.0, max=MAX_THRUST_PER_MOTOR_N)


def induced_electrical_power(thrusts: torch.Tensor) -> torch.Tensor:
    """Per-motor electrical power draw from thrust via actuator-disk
    (momentum theory) induced power, corrected by ROTOR_FIGURE_OF_MERIT.

    thrusts: (num_envs, 4) newtons, each >= 0
    returns: (num_envs,) total electrical power in watts across all 4 motors

    P_ideal = sqrt(T^3 / (2 * rho * A))   [momentum theory]
    P_elec  = P_ideal / FM

    See DRONE_SPEC.md section 3 and
    https://en.wikipedia.org/wiki/Momentum_theory
    """
    thrusts_clamped = thrusts.clamp(min=0.0)
    power_per_motor = C_POWER_ELECTRICAL * torch.pow(thrusts_clamped, 1.5)
    return power_per_motor.sum(dim=-1)
