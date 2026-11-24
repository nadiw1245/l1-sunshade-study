"""Phase 1 validation: CR3BP, L1, solar radiation pressure, Lyapunov orbit, station keeping."""
import numpy as np
import pytest

from src import constants as k
from src.dynamics import cr3bp, stationkeeping as sk

S = cr3bp.DEFAULT
MU = S.mu


# ---- required validation --------------------------------------------------------------
def test_classical_l1_is_1p5_million_km_from_earth():
    """REQUIRED: classical L1 distance ~ 1.5 million km (within 0.5 %)."""
    d_km = cr3bp.l1_distance_from_secondary(S) / 1e3
    assert d_km == pytest.approx(1.5e6, rel=0.005)


def test_jacobi_constant_conserved_over_one_year_without_srp():
    """REQUIRED: Jacobi constant conserved to < 1e-8 over one simulated year (no SRP)."""
    x_l1 = cr3bp.l1_position(MU)
    # a generic, non-periodic 3-D trajectory starting 100,000 km from L1
    s0 = np.array([x_l1 + 1e8 / S.length, 2e7 / S.length, 5e7 / S.length, 1e-4, -2e-4, 5e-5])
    t = np.linspace(0, S.year, 2000)
    sol = cr3bp.propagate(s0, (0, S.year), MU, 0.0, t_eval=t)
    assert sol.success and sol.t[-1] == pytest.approx(2 * np.pi)
    c = cr3bp.jacobi_constant(sol.y, MU)
    assert np.max(np.abs(c - c[0])) < 1e-8
    assert np.max(np.abs(c - c[0])) < 1e-11       # what DOP853 at rtol=1e-12 actually achieves


def test_jacobi_constant_conserved_with_constant_srp():
    beta = 0.03
    x_eq = cr3bp.l1_position(MU, beta)
    s0 = np.array([x_eq + 5e-4, 1e-4, 0, 0, -1e-3, 0])
    sol = cr3bp.propagate(s0, (0, S.year), MU, beta, t_eval=np.linspace(0, S.year, 500))
    c = cr3bp.jacobi_constant(sol.y, MU, beta)
    assert np.max(np.abs(c - c[0])) < 1e-8


# ---- L1 ------------------------------------------------------------------------------
def test_l1_is_an_equilibrium_and_solvers_agree():
    x = cr3bp.l1_position(MU)
    acc = cr3bp.acceleration([x, 0, 0, 0, 0, 0], MU)
    assert np.max(np.abs(acc)) < 1e-12
    assert cr3bp.l1_position_newton(MU) == pytest.approx(x, abs=1e-13)


def test_l1_matches_hill_series():
    """Independent check: gamma = r_H (1 - r_H/3 - r_H^2/9), r_H = (mu/3)^(1/3)."""
    rh = (MU / 3) ** (1 / 3)
    series = rh * (1 - rh / 3 - rh**2 / 9)
    assert (1 - MU - cr3bp.l1_position(MU)) == pytest.approx(series, rel=1e-5)


def test_earth_only_l1_is_slightly_closer():
    d_e = cr3bp.l1_distance_from_secondary(cr3bp.SUN_EARTH)
    d_em = cr3bp.l1_distance_from_secondary(cr3bp.SUN_EARTH_MOON)
    assert 1.49e9 < d_e < d_em < 1.50e9


def test_units():
    assert S.time * 2 * np.pi / k.DAY == pytest.approx(365.25, rel=2e-4)   # one year
    assert S.velocity == pytest.approx(29.78e3, rel=1e-3)                  # Earth's orbital speed


# ---- solar radiation pressure -----------------------------------------------------------
def test_critical_areal_density():
    assert k.SIGMA_STAR * 1e3 == pytest.approx(1.53, abs=0.01)             # g/m^2
    assert cr3bp.lightness_number(k.SIGMA_STAR, 1.0) == pytest.approx(1.0)
    assert cr3bp.lightness_number(0.03, 0.08) == pytest.approx(0.08 * k.SIGMA_STAR / 0.03)


def test_equilibrium_moves_sunward_monotonically_with_beta():
    betas = np.linspace(0, 0.2, 30)
    d = np.array([cr3bp.l1_distance_from_secondary(S, b) for b in betas])
    assert np.all(np.diff(d) > 0)
    for b in (0.01, 0.05):
        x = cr3bp.l1_position(MU, b)
        assert np.max(np.abs(cr3bp.acceleration([x, 0, 0, 0, 0, 0], MU, b))) < 1e-12


def test_beta_for_distance_inverts_equilibrium():
    for b in (0.0, 0.004, 0.03, 0.2):
        d = cr3bp.l1_distance_from_secondary(S, b)
        assert cr3bp.beta_for_distance(d, S) == pytest.approx(b, abs=1e-12)


def test_no_equilibrium_when_radiation_pressure_exceeds_gravity():
    with pytest.raises(ValueError):
        cr3bp.l1_position(MU, 1.2)


@pytest.mark.parametrize("q, sigma_g, angel_gm, tol", [
    (0.08, 30.0, 1.58, 0.02),    # Early's glass screen as quoted by Angel (2006)
    (1.00, 40.0, 2.50, 0.02),    # fully reflective shade ("McInnes" point in Angel Fig. 2b)
    (0.10, 5.6, 2.25, 0.07),     # read from Angel's Fig. 2
])
def test_equilibrium_distance_matches_angel_2006(q, sigma_g, angel_gm, tol):
    beta = cr3bp.lightness_number(sigma_g * 1e-3, q)
    d_gm = cr3bp.l1_distance_from_secondary(S, beta) / 1e9
    assert d_gm == pytest.approx(angel_gm, rel=tol)


# ---- instability ----------------------------------------------------------------------
def test_l1_is_a_saddle_times_centre():
    lin = cr3bp.linearise(MU)
    eig = np.linalg.eigvals(lin.a_planar)
    assert np.sum(eig.real > 1e-6) == 1 and np.sum(eig.real < -1e-6) == 1
    assert np.sum(np.abs(eig.real) < 1e-9) == 2
    assert 20 < cr3bp.e_folding_time_days(S) < 26


def test_small_offset_grows_exponentially_at_linear_rate():
    t, sep, _ = cr3bp.divergence_from_equilibrium(1e3, S, days=250)
    assert sep[0] == pytest.approx(1e3, rel=1e-6)
    assert sep[-1] > 1e3 * 1e3                      # 1 km became > 1000 km
    late = (t > 120) & (t < 220)                    # unstable mode dominates, still linear regime
    rate = np.polyfit(t[late], np.log(sep[late]), 1)[0]
    assert 1 / rate == pytest.approx(cr3bp.e_folding_time_days(S), rel=0.03)


# ---- Lyapunov orbit --------------------------------------------------------------------
@pytest.fixture(scope="module")
def orbit():
    return cr3bp.lyapunov_orbit(2e8 / S.length, MU)


def test_lyapunov_orbit_closes(orbit):
    assert orbit.residual < 1e-11
    sol = cr3bp.propagate(orbit.state0, (0, orbit.period), MU)
    assert np.max(np.abs(sol.y[:, -1] - orbit.state0)) < 1e-9
    assert np.max(np.abs(sol.y[:, -1] - orbit.state0)) * S.length < 1e3   # closes to < 1 km


def test_lyapunov_period_and_stability(orbit):
    period_days = orbit.period * S.time / k.DAY
    lin = cr3bp.linearise(MU)
    assert period_days == pytest.approx(2 * np.pi / lin.omega * S.time / k.DAY, rel=0.02)
    assert 170 < period_days < 185                   # ~ half a year, like real L1 missions
    eig = np.linalg.eigvals(orbit.monodromy)
    assert np.prod(eig).real == pytest.approx(1.0, rel=1e-6)   # symplectic map
    assert orbit.stability_multiplier > 1000         # strongly unstable


def test_lyapunov_family_and_srp_orbit():
    amps = np.array([0.5e8, 1e8, 2e8, 4e8]) / S.length
    fam = cr3bp.lyapunov_family(amps, MU)
    periods = [o.period for o in fam]
    assert all(o.residual < 1e-10 for o in fam)
    assert np.all(np.diff(periods) > 0)              # period grows with amplitude
    beta = cr3bp.lightness_number(5.6e-3, 0.1)
    o = cr3bp.lyapunov_orbit(2e8 / S.length, MU, beta)
    assert o.residual < 1e-10


# ---- station keeping -------------------------------------------------------------------
def test_station_keeping_keeps_shade_near_l1_nonlinear():
    r = sk.simulate_impulsive(interval_days=14, n_trials=3, nonlinear=True, seed=1)
    assert r.max_excursion < 100e3                   # stays within 100 km for a year
    assert 0.005 < r.dv_per_year < 1.0               # m/s per year with 1 km / 1 mm/s navigation


def test_linear_and_nonlinear_station_keeping_agree():
    kw = dict(interval_days=14, beta=0.027, beta_noise=0.01)
    lin = sk.simulate_impulsive(n_trials=200, **kw)
    non = sk.simulate_impulsive(n_trials=6, nonlinear=True, **kw)
    assert non.dv_per_year == pytest.approx(lin.dv_per_year, rel=0.35)


def test_radiation_pressure_uncertainty_dominates_dv():
    quiet = sk.simulate_impulsive(beta=0.027, beta_noise=0.0)
    noisy = sk.simulate_impulsive(beta=0.027, beta_noise=0.01)
    assert noisy.dv_per_year > 100 * quiet.dv_per_year


def test_delta_beta_control_is_small():
    db = sk.delta_beta_to_hold_offset(1e6)           # 1000 km offset
    assert 1e-5 < db < 1e-3
    assert sk.solar_gravity_at(S) == pytest.approx(k.GM_SUN / (k.AU - 1.4976e9) ** 2, rel=1e-3)
