"""Phase 3 validation: energy balance models."""
import numpy as np
import pytest

from src import constants as k
from src.climate import ebm, latitude, scenarios


def test_effective_temperature_is_255_k_without_greenhouse():
    """REQUIRED: ~255 K effective temperature with no greenhouse forcing."""
    assert ebm.effective_temperature() == pytest.approx(255.0, abs=1.0)
    bare = ebm.EnergyBalanceModel(emissivity=1.0)            # eps = 1, F_CO2 = 0
    assert bare.equilibrium_temperature() == pytest.approx(255.0, abs=1.0)
    # and the time-dependent model relaxes to it from any start
    t, temp = bare.integrate(60.0, lambda t: bare.co2_ref, t0=300.0)
    assert temp[-1] == pytest.approx(ebm.effective_temperature(), abs=1e-3)


def test_co2_forcing():
    assert ebm.co2_forcing(560, 280) == pytest.approx(3.708, abs=1e-3)
    assert ebm.co2_forcing(280, 280) == 0.0


def test_calibration_reproduces_present_day_temperature():
    m = ebm.calibrated_model()
    assert m.equilibrium_temperature() == pytest.approx(288.0, abs=1e-6)
    assert 0.55 < m.emissivity < 0.65
    assert abs(float(m.net_flux(288.0))) < 1e-9


def test_planck_only_sensitivity_and_optional_feedback():
    m = ebm.calibrated_model()
    assert m.climate_sensitivity() == pytest.approx(1.1, abs=0.1)
    assert ebm.calibrated_model(sensitivity=3.0).climate_sensitivity() == pytest.approx(3.0, abs=0.2)


def test_shade_fraction_cancels_doubling_exactly():
    m = ebm.calibrated_model()
    f = m.blocked_fraction_to_cancel(560.0, 280.0)
    assert f == pytest.approx(4 * 3.708 / (k.S0 * 0.7), rel=1e-3)
    assert 0.014 < f < 0.02                                   # cf. 1.7-1.8 % in the literature
    assert m.equilibrium_temperature(f, 560.0) == pytest.approx(m.equilibrium_temperature(0.0, 280.0), abs=1e-9)
    # independent of the feedback strength
    assert ebm.calibrated_model(sensitivity=3.0).blocked_fraction_to_cancel(560.0, 280.0) == pytest.approx(f)


def test_integration_matches_analytic_linear_relaxation():
    m = ebm.calibrated_model()
    tau = m.response_time_years()
    t, temp = m.integrate(3 * tau, lambda t: m.co2_ref, t0=288.5, n_out=301)
    expected = 288.0 + 0.5 * np.exp(-t / tau)
    assert np.max(np.abs(temp - expected)) < 2e-3


@pytest.fixture(scope="module")
def runs():
    return scenarios.run_scenarios()


def test_scenarios_behave_physically(runs):
    a, b, c, d = runs.values()
    m = ebm.calibrated_model()
    assert a.temp[0] == pytest.approx(288.0, abs=1e-6)
    assert a.temp[-1] == pytest.approx(m.equilibrium_temperature(co2=560.0), abs=1e-3)
    pre = m.equilibrium_temperature(co2=280.0)
    assert b.temp[-1] == pytest.approx(pre, abs=1e-3)         # doubling cancelled
    assert c.temp[-1] == pytest.approx(pre, abs=1e-3)
    assert d.temp[-1] == pytest.approx(a.temp[-1], abs=1e-3)  # after failure: back to no-shade
    before_fail = d.t < 100
    assert np.allclose(d.temp[before_fail], c.temp[before_fail], atol=1e-9)


def test_termination_shock_warms_much_faster_than_no_shade(runs):
    a, _, _, d = runs.values()
    assert scenarios.max_warming_rate(d) > 5 * scenarios.max_warming_rate(a)


def test_annual_insolation_distribution():
    x = np.linspace(-1, 1, 4001)
    s = latitude.annual_insolation_distribution(x)
    assert np.trapezoid(s, x) / 2 == pytest.approx(1.0, abs=1e-4)          # global mean
    s2 = 5 * np.trapezoid(s * latitude.p2(x), x) / 2
    assert s2 == pytest.approx(-0.477, abs=0.005)                         # classic Legendre coefficient
    assert s[2000] > 1.2 and s[0] < 0.55                                  # equator bright, poles dim


def test_latitude_model_calibration_and_energy_balance():
    m = latitude.calibrated_latitude_model()
    t = m.solve()
    assert m.global_mean(t) == pytest.approx(288.0, abs=1e-6)
    assert 25 < t.max() - t.min() < 60
    absorbed = np.mean(m.s0 / 4 * m.insolation * (1 - m.albedo))
    emitted = np.mean(m.emissivity * k.SIGMA_SB * t**4)
    assert absorbed == pytest.approx(emitted, rel=1e-9)                   # transport only redistributes
    assert absorbed == pytest.approx(k.S0 * 0.7 / 4, rel=1e-9)


def test_shade_does_not_cancel_co2_at_every_latitude():
    m = latitude.calibrated_latitude_model()
    base = m.solve()
    co2 = 2 * m.co2_ref
    f = m.shade_fraction_restoring_global_mean(co2)
    resid = m.solve(f, co2) - base
    assert abs(resid.mean()) < 1e-8                 # global mean restored ...
    assert resid[0] > 0.1 and resid[-1] > 0.1       # ... but the poles stay warmer
    assert resid[m.n // 2] < -0.05                  # ... and the tropics are over-cooled
