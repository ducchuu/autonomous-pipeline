"""Vectorized LiPo battery discharge / voltage-sag model for the drone env.

No isaaclab dependency -- pure torch, so it is unit-testable in isolation
(see tests/test_battery.py). Every constant is documented in DRONE_SPEC.md
section 4.

Model summary
-------------
1. Coulomb counting: integrate instantaneous current draw (electrical power
   / current terminal voltage) into consumed charge, tracked as state of
   charge (SoC) in [0, 1].
2. Open-circuit voltage (OCV) is a piecewise-linear function of SoC, from a
   published LiPo discharge table (see OCV_SOC_TABLE below).
3. Terminal voltage under load = OCV(SoC) - I_total * R_internal (first-order
   Thevenin/resistor-only model -- good enough for an RL efficiency signal,
   not meant to capture RC relaxation dynamics).
4. "Battery dead" is True when SoC <= 0 OR terminal voltage drops below the
   3.5 V/cell safety cutoff under the current load, matching real flight
   controller low-voltage cutoff behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# --------------------------------------------------------------------------
# Named constants -- see DRONE_SPEC.md section 4
# --------------------------------------------------------------------------

CELL_COUNT = 4                       # [MEASURED] 4S
CAPACITY_MAH = 1550.0                # [MEASURED]
C_RATING = 100.0                     # [MEASURED]
MAX_CONTINUOUS_CURRENT_A = CAPACITY_MAH / 1000.0 * C_RATING  # [DERIVED] = 155 A

FULL_CHARGE_V_PER_CELL = 4.20        # [standard LiPo spec]
NOMINAL_V_PER_CELL = 3.70            # [standard LiPo spec]
CUTOFF_V_PER_CELL = 3.50             # [standard safe-cutoff convention, see DRONE_SPEC.md]

INTERNAL_RESISTANCE_OHM = 0.012      # [ESTIMATE-VERIFY] ~3 mOhm/cell x 4 cells in series

# Single-cell OCV (V) vs state of charge (fraction), piecewise-linear.
# Source: https://lionbattery.in/battery-voltage-chart/ (LiPo table).
_SOC_BREAKPOINTS = torch.tensor(
    [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
)
_CELL_VOLTAGE_BREAKPOINTS = torch.tensor(
    [3.00, 3.60, 3.74, 3.77, 3.79, 3.82, 3.87, 3.92, 4.00, 4.10, 4.20]
)


def _interp_cell_ocv(soc: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of single-cell OCV at the given SoC
    tensor (any shape, values expected in [0, 1] but clamped defensively).
    """
    soc_clamped = soc.clamp(0.0, 1.0)
    breakpoints = _SOC_BREAKPOINTS.to(soc.device)
    voltages = _CELL_VOLTAGE_BREAKPOINTS.to(soc.device)
    # torch has no built-in 1D interp; implement via searchsorted + lerp.
    idx = torch.searchsorted(breakpoints, soc_clamped, right=True).clamp(1, len(breakpoints) - 1)
    x0 = breakpoints[idx - 1]
    x1 = breakpoints[idx]
    y0 = voltages[idx - 1]
    y1 = voltages[idx]
    t = (soc_clamped - x0) / (x1 - x0).clamp(min=1e-9)
    return y0 + t * (y1 - y0)


@dataclass
class BatteryModel:
    """Per-environment vectorized battery state.

    All tensors have shape (num_envs,). Call :meth:`reset` for the
    environments being reset and :meth:`step` once per physics/control step
    with the total electrical power draw for that step.
    """

    num_envs: int
    device: torch.device

    def __post_init__(self) -> None:
        self.consumed_mah = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.consumed_mah[env_ids] = 0.0

    @property
    def capacity_mah(self) -> float:
        return CAPACITY_MAH

    def state_of_charge(self) -> torch.Tensor:
        """Return SoC in [0, 1] (clamped) per environment."""
        return (1.0 - self.consumed_mah / CAPACITY_MAH).clamp(0.0, 1.0)

    def open_circuit_voltage(self) -> torch.Tensor:
        """Pack-level open-circuit voltage (no load) per environment."""
        soc = self.state_of_charge()
        return CELL_COUNT * _interp_cell_ocv(soc)

    def step(self, electrical_power_w: torch.Tensor, dt_s: float) -> torch.Tensor:
        """Advance the battery state by one step given total instantaneous
        electrical power draw (watts) for each environment.

        Uses the *previous* step's terminal voltage to convert power to
        current (avoids an implicit/circular solve; error is negligible at
        typical RL control rates of tens of Hz).

        Returns the new terminal voltage tensor (num_envs,).
        """
        ocv = self.open_circuit_voltage()
        # First estimate of current using OCV (no-load) voltage to avoid
        # division by a not-yet-known sagged voltage.
        current_estimate_a = electrical_power_w / ocv.clamp(min=1e-6)
        terminal_voltage = ocv - current_estimate_a * INTERNAL_RESISTANCE_OHM
        terminal_voltage = terminal_voltage.clamp(min=0.1)

        # Recompute current at the (now known) sagged terminal voltage for
        # the coulomb-counting integral -- one fixed-point refinement step.
        current_a = electrical_power_w / terminal_voltage
        delta_mah = current_a * (dt_s / 3600.0) * 1000.0
        self.consumed_mah = self.consumed_mah + delta_mah

        return terminal_voltage

    def is_dead(self, terminal_voltage: torch.Tensor) -> torch.Tensor:
        """Boolean tensor: True where the battery should be considered dead
        this step (SoC exhausted OR sagged below the safe cutoff voltage).
        """
        soc_dead = self.state_of_charge() <= 0.0
        voltage_dead = terminal_voltage < (CELL_COUNT * CUTOFF_V_PER_CELL)
        return soc_dead | voltage_dead
