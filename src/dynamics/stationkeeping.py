"""Station keeping at the (SRP-displaced) L1 equilibrium.

The in-plane linearised motion about the equilibrium has one unstable mode (growth
rate lam), one stable mode and one oscillatory (centre) mode.  Only the unstable
mode has to be controlled.  Writing delta = [dx, dy, dvx, dvy] and w for the left
eigenvector of the unstable mode (normalised so that w.u = 1), the unstable amplitude
is alpha = w . delta and it grows as exp(lam t).

Impulsive strategy
    Every `interval` days the shade measures its state (with navigation error) and
    fires the smallest impulse dv that zeroes the *estimated* alpha:

        dv = -alpha_est * w_v / |w_v|^2

    The yearly cost is therefore set by how well alpha can be measured and by how
    long errors are left to grow, not by the size of the shade.

Radiation-pressure strategy
    Changing the lightness number by d_beta changes the radial solar force by
    d_beta * GM_sun / r^2, which is a control acceleration along the Sun line obtained
    without propellant (Angel 2006 proposes exactly this).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from src import constants as k
from src.dynamics import cr3bp


@dataclass
class StationKeepingResult:
    dv_per_year: float          # m/s per year (mean over trials)
    dv_per_year_std: float
    dv_per_burn: float          # m/s (mean)
    max_excursion: float        # m, largest distance from the equilibrium (mean over trials)
    interval_days: float
    n_trials: int


def simulate_impulsive(system: cr3bp.System = cr3bp.DEFAULT, beta: float = 0.0, *,
                       interval_days: float = 14.0, sigma_pos: float = 1.0e3,
                       sigma_vel: float = 1.0e-3, exec_error: float = 0.01,
                       beta_noise: float = 0.0, years: float = 1.0, n_trials: int = 200,
                       seed: int = 0, nonlinear: bool = False) -> StationKeepingResult:
    """Monte-Carlo estimate of the impulsive station-keeping budget.

    sigma_pos, sigma_vel : 1-sigma orbit-determination errors per axis [m], [m/s]
    exec_error           : 1-sigma fractional burn magnitude error
    beta_noise           : 1-sigma fractional error in the shade's lightness number,
                           redrawn every interval (unmodelled change in reflectivity,
                           attitude or solar output).  It acts as an unmodelled radial
                           acceleration  beta * beta_noise * g_sun.
    nonlinear            : propagate with the full CR3BP (DOP853) instead of the
                           linearised flow; used to verify the linear model.
    """
    rng = np.random.default_rng(seed)
    mu = system.mu
    lin = cr3bp.linearise(mu, beta)
    dt = interval_days * k.DAY / system.time
    phi = expm(lin.a_planar * dt)
    # Response of the state after dt to a constant unit acceleration along +x
    # (away from the Sun), from the augmented-matrix exponential.
    aug = np.zeros((5, 5))
    aug[:4, :4] = lin.a_planar
    aug[2, 4] = 1.0
    gamma = expm(aug * dt)[:4, 4]
    g_sun = (1 - mu) / (lin.x_eq + mu) ** 2
    w = lin.unstable_left
    w_v = w[2:]
    n_burns = int(round(years * 365.25 / interval_days))
    sp = sigma_pos / system.length
    sv = sigma_vel / system.velocity

    totals, excursions, burns = [], [], []
    for _ in range(n_trials):
        delta = np.concatenate([rng.normal(0, sp, 2), rng.normal(0, sv, 2)])
        total, worst = 0.0, 0.0
        for _ in range(n_burns):
            est = delta + np.concatenate([rng.normal(0, sp, 2), rng.normal(0, sv, 2)])
            dv = -(w @ est) * w_v / (w_v @ w_v)
            total += np.linalg.norm(dv)
            burns.append(np.linalg.norm(dv))
            delta = delta.copy()
            delta[2:] += dv * (1 + rng.normal(0, exec_error))
            d_beta = beta * beta_noise * rng.normal()
            if nonlinear:
                s0 = np.array([lin.x_eq + delta[0], delta[1], 0, delta[2], delta[3], 0])
                sol = cr3bp.propagate(s0, (0, dt), mu, beta + d_beta)
                sf = sol.y[:, -1]
                delta = np.array([sf[0] - lin.x_eq, sf[1], sf[3], sf[4]])
                worst = max(worst, np.max(np.hypot(sol.y[0] - lin.x_eq, sol.y[1])))
            else:
                delta = phi @ delta + gamma * d_beta * g_sun
                worst = max(worst, np.hypot(delta[0], delta[1]))
        totals.append(total / years)
        excursions.append(worst)
    totals = np.array(totals) * system.velocity
    return StationKeepingResult(float(totals.mean()), float(totals.std()),
                                float(np.mean(burns) * system.velocity),
                                float(np.mean(excursions) * system.length),
                                interval_days, n_trials)


def uncontrolled_growth_factor(days: float, system: cr3bp.System = cr3bp.DEFAULT,
                               beta: float = 0.0) -> float:
    """exp(lam t): amplification of an unstable-mode error left alone for `days`."""
    lin = cr3bp.linearise(system.mu, beta)
    return float(np.exp(lin.lam * days * k.DAY / system.time))


def solar_gravity_at(system: cr3bp.System, beta: float = 0.0) -> float:
    """Solar gravitational acceleration [m/s^2] at the equilibrium point."""
    mu = system.mu
    x = cr3bp.l1_position(mu, beta)
    return (1 - mu) / (x + mu) ** 2 * system.accel


def delta_beta_to_hold_offset(offset_m: float, system: cr3bp.System = cr3bp.DEFAULT,
                              beta: float = 0.0) -> float:
    """Change in lightness number that statically balances an along-axis offset.

    A shade displaced by dx along the Sun line feels a net acceleration U_xx * dx
    pushing it further away.  Modulating the radiation force by d_beta * g_sun
    cancels it when  d_beta = U_xx * dx / g_sun  (all non-dimensional).
    """
    mu = system.mu
    x = cr3bp.l1_position(mu, beta)
    uxx = cr3bp.potential_hessian([x, 0, 0], mu, beta)[0, 0]
    g_sun = (1 - mu) / (x + mu) ** 2
    return float(uxx * (offset_m / system.length) / g_sun)


def control_acceleration_from_delta_beta(d_beta: float, system: cr3bp.System = cr3bp.DEFAULT,
                                         beta: float = 0.0) -> float:
    """Control acceleration [m/s^2] available from modulating beta by d_beta."""
    return d_beta * solar_gravity_at(system, beta)


def propellant_fraction(dv_per_year: float, years: float, isp: float) -> float:
    """Tsiolkovsky propellant mass fraction for a total dv over the mission."""
    g0 = 9.80665
    return float(1 - np.exp(-dv_per_year * years / (isp * g0)))
