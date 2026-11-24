"""The four shade scenarios of the study, run with the zero-dimensional EBM.

Common CO2 pathway (an idealised assumption, not a projection): CO2 starts at the
present-day 420 ppm, grows exponentially at `growth` per year until it reaches twice
the pre-industrial concentration (560 ppm) and is then held constant.

(a) no shade
(b) full shade: a shade sized to cancel the forcing of the whole 280 -> 560 ppm
    doubling is switched on at once at `t_deploy`
(c) gradual shade: the same shade is built up linearly over `ramp_years`
(d) termination shock: as (c), but the shade is lost instantly at `t_fail`
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.climate.ebm import CO2_PREINDUSTRIAL, CO2_PRESENT, EnergyBalanceModel, calibrated_model


@dataclass
class ScenarioResult:
    name: str
    t: np.ndarray       # years
    temp: np.ndarray    # K
    f: np.ndarray       # blocked fraction
    co2: np.ndarray     # ppm


def co2_pathway(growth: float = 0.005, start: float = CO2_PRESENT,
                cap: float = 2 * CO2_PREINDUSTRIAL):
    t_cap = np.log(cap / start) / np.log1p(growth)

    def co2(t):
        return float(min(start * (1 + growth) ** t, cap))

    return co2, float(t_cap)


def run_scenarios(model: EnergyBalanceModel | None = None, *, years: float = 150.0,
                  growth: float = 0.005, t_deploy: float = 30.0, ramp_years: float = 20.0,
                  t_fail: float = 100.0, n_out: int = 1501) -> dict[str, ScenarioResult]:
    model = model or calibrated_model()
    co2, t_cap = co2_pathway(growth)
    f_full = model.blocked_fraction_to_cancel(2 * CO2_PREINDUSTRIAL, CO2_PREINDUSTRIAL)

    def ramp(t):
        return float(f_full * np.clip((t - t_deploy) / ramp_years, 0.0, 1.0))

    shades = {
        "a) no shade": lambda t: 0.0,
        "b) full shade switched on": lambda t: f_full if t >= t_deploy else 0.0,
        "c) shade deployed over 20 yr": ramp,
        "d) shade fails (termination shock)": lambda t: ramp(t) if t < t_fail else 0.0,
    }
    breaks = (t_cap, t_deploy, t_deploy + ramp_years, t_fail)
    out = {}
    for name, f_of_t in shades.items():
        t, temp = model.integrate(years, co2, f_of_t, n_out=n_out, breakpoints=breaks)
        out[name] = ScenarioResult(name, t, temp, np.array([f_of_t(x) for x in t]),
                                   np.array([co2(x) for x in t]))
    return out


def max_warming_rate(result: ScenarioResult, window_years: float = 10.0) -> float:
    """Largest warming over any `window_years` window, in K per decade."""
    dt = result.t[1] - result.t[0]
    n = int(round(window_years / dt))
    return float(np.max(result.temp[n:] - result.temp[:-n]) * 10.0 / window_years)
