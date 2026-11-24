"""Zero-dimensional global energy balance model (EBM).

    C dT/dt = S (1 - f) (1 - alpha) / 4  -  eps * sigma_SB * T^4  +  F_CO2
    F_CO2   = 5.35 ln(CO2 / CO2_ref)   [W/m^2]   (Myhre et al. 1998 simplified expression)

T      global mean surface temperature [K]
S      total solar irradiance [W/m^2]
f      fraction of sunlight blocked by the shade
alpha  planetary (Bond) albedo
eps    effective emissivity of the Earth seen from space (< 1 because of the greenhouse
       effect); the single calibrated parameter
C      heat capacity per unit area of the part of the climate system that follows
       the surface temperature on decadal time scales (ocean mixed layer)

Conventions
-----------
The model is calibrated so that its equilibrium at the *reference* CO2 concentration
(present day, 420 ppm) is the present-day global mean temperature (288 K).  F_CO2 is
measured relative to that reference state, so the pre-industrial 280 ppm appears as a
negative forcing.

Honest caveat: the only feedback in this model is the Planck (T^4) response, so its
climate sensitivity (~1.1 K per CO2 doubling) is well below the ~3 K of comprehensive
models.  `feedback` [W/m^2/K] is an optional extra term  +feedback*(T - T_ref)  that
represents the net effect of water-vapour, lapse-rate, cloud and ice feedbacks and can
be set to reproduce a chosen sensitivity.  It is 0 by default (the equation above).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from src import constants as k

SECONDS_PER_YEAR = k.YEAR
CO2_PREINDUSTRIAL = 280.0    # ppm
CO2_PRESENT = 420.0          # ppm
T_PRESENT = 288.0            # K, present-day global mean surface temperature (~15 C)
ALBEDO = 0.30
# Ocean mixed layer: 70 % ocean cover * density * specific heat * 70 m depth
HEAT_CAPACITY = 0.7 * 1025.0 * 3990.0 * 70.0   # J m^-2 K^-1  (~2.0e8)


def co2_forcing(co2_ppm, co2_ref: float = CO2_PREINDUSTRIAL):
    """Radiative forcing of CO2 relative to co2_ref [W/m^2]."""
    return 5.35 * np.log(np.asarray(co2_ppm, float) / co2_ref)


F_2XCO2 = float(co2_forcing(2.0, 1.0))   # 3.708 W/m^2


def effective_temperature(s: float = k.S0, albedo: float = ALBEDO, f: float = 0.0) -> float:
    """Equilibrium temperature with no greenhouse effect (eps = 1, F = 0)."""
    return (s * (1 - f) * (1 - albedo) / (4 * k.SIGMA_SB)) ** 0.25


@dataclass(frozen=True)
class EnergyBalanceModel:
    s: float = k.S0
    albedo: float = ALBEDO
    emissivity: float = 1.0
    heat_capacity: float = HEAT_CAPACITY
    co2_ref: float = CO2_PRESENT
    t_ref: float = T_PRESENT
    feedback: float = 0.0          # W/m^2/K, optional (see module docstring)

    # ---- fluxes ----------------------------------------------------------------------
    def absorbed_solar(self, f=0.0):
        return self.s * (1 - np.asarray(f, float)) * (1 - self.albedo) / 4

    def outgoing_longwave(self, t):
        return self.emissivity * k.SIGMA_SB * np.asarray(t, float) ** 4

    def net_flux(self, t, f=0.0, co2=None):
        """Top-of-atmosphere imbalance N [W/m^2] (positive = planet gains heat)."""
        co2 = self.co2_ref if co2 is None else co2
        return (self.absorbed_solar(f) - self.outgoing_longwave(t)
                + co2_forcing(co2, self.co2_ref) + self.feedback * (np.asarray(t, float) - self.t_ref))

    # ---- equilibrium -----------------------------------------------------------------
    def equilibrium_temperature(self, f: float = 0.0, co2: float | None = None) -> float:
        # With the optional linear feedback the balance is only stable (dN/dT < 0) above
        # t_min, so the root is searched on the stable branch.
        t_min = 150.0
        if self.feedback > 0:
            t_min = max(t_min, 1.001 * (self.feedback / (4 * self.emissivity * k.SIGMA_SB)) ** (1 / 3))
        return brentq(lambda t: float(self.net_flux(t, f, co2)), t_min, 400.0, xtol=1e-12)

    def restoring_strength(self, t: float | None = None) -> float:
        """-dN/dT [W/m^2/K]: how strongly the planet resists a temperature change."""
        t = self.t_ref if t is None else t
        return 4 * self.emissivity * k.SIGMA_SB * t**3 - self.feedback

    def climate_sensitivity(self) -> float:
        """Equilibrium warming for a doubling of CO2 from the reference state [K]."""
        return self.equilibrium_temperature(co2=2 * self.co2_ref) - self.equilibrium_temperature()

    def response_time_years(self) -> float:
        return self.heat_capacity / self.restoring_strength() / SECONDS_PER_YEAR

    def blocked_fraction_to_cancel(self, co2: float, co2_from: float | None = None) -> float:
        """Shade fraction f that cancels the forcing of raising CO2 from co2_from to co2.

        From  S f (1 - alpha) / 4 = 5.35 ln(co2/co2_from).  Independent of emissivity,
        feedback and heat capacity.
        """
        co2_from = self.co2_ref if co2_from is None else co2_from
        return float(4 * co2_forcing(co2, co2_from) / (self.s * (1 - self.albedo)))

    # ---- time integration ------------------------------------------------------------
    def integrate(self, years: float, co2_of_t: Callable[[float], float],
                  f_of_t: Callable[[float], float] = lambda t: 0.0, t0: float | None = None,
                  n_out: int = 1001, breakpoints=()):
        """Integrate T(t) for `years` years.  co2_of_t and f_of_t take time in years.

        The integration is restarted at each breakpoint (times where the forcing has a
        kink or jump) so that the adaptive integrator never steps across a discontinuity.
        """
        t_init = self.equilibrium_temperature(f_of_t(0.0), co2_of_t(0.0)) if t0 is None else t0

        def rhs(t, y):
            return self.net_flux(y, f_of_t(t), co2_of_t(t)) * SECONDS_PER_YEAR / self.heat_capacity

        t_out = np.linspace(0.0, years, n_out)
        edges = sorted({0.0, float(years), *[float(b) for b in breakpoints if 0 < b < years]})
        temps = np.empty_like(t_out)
        y = np.array([t_init])
        for a, b in zip(edges[:-1], edges[1:]):
            mask = (t_out >= a) & (t_out <= b)
            sol = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=1e-10, atol=1e-12,
                            dense_output=True)
            temps[mask] = sol.sol(t_out[mask])[0]
            y = sol.y[:, -1]
        return t_out, temps


def calibrated_model(t_present: float = T_PRESENT, co2_present: float = CO2_PRESENT,
                     sensitivity: float | None = None, **kwargs) -> EnergyBalanceModel:
    """Model whose equilibrium at present-day CO2 is the present-day temperature.

    With f = 0 and F_CO2 = 0 at the reference state the balance is
    S (1 - alpha)/4 = eps sigma T^4, which fixes eps.
    If `sensitivity` [K per CO2 doubling] is given, the optional feedback term is set
    so that the linearised model has that equilibrium climate sensitivity.
    """
    base = EnergyBalanceModel(co2_ref=co2_present, t_ref=t_present, **kwargs)
    eps = float(base.absorbed_solar(0.0) / (k.SIGMA_SB * t_present**4))
    model = replace(base, emissivity=eps)
    if sensitivity is not None:
        planck = 4 * eps * k.SIGMA_SB * t_present**3
        model = replace(model, feedback=planck - F_2XCO2 / sensitivity)
    return model
