"""Phase 4 validation: mass and launch budget."""
import numpy as np
import pytest

from src.engineering import budget


@pytest.mark.parametrize("q, sigma_g, angel_mt, tol", [
    (0.08, 30.0, 100.0, 0.10),   # Early's glass screen
    (0.10, 5.6, 27.0, 0.10),
    (0.032, 2.5, 11.0, 0.10),
    (1.00, 40.0, 270.0, 0.10),   # fully reflective foil
])
def test_total_mass_matches_angel_2006(q, sigma_g, angel_mt, tol):
    d = budget.design(0.018, sigma_g * 1e-3, q)
    assert d.mass / 1e9 == pytest.approx(angel_mt, rel=tol)     # million tonnes


def test_mass_is_area_times_density():
    d = budget.design(0.018, 5.6e-3, 0.1)
    assert d.mass == pytest.approx(d.area * 5.6e-3)
    assert budget.shade_mass(1e12, 0.01) == 1e10


def test_lighter_is_not_always_better_for_a_reflective_shade():
    """A reflective shade that is too light is pushed far sunward and needs more area."""
    heavy = budget.design(0.018, 40e-3, 1.0)
    light = budget.design(0.018, 5e-3, 1.0)
    assert light.distance > heavy.distance
    assert light.mass > heavy.mass
    with pytest.raises(ValueError):
        budget.design(0.018, 1e-3, 1.0)       # beta > 1: blown away, no equilibrium


def test_launch_counts():
    v = budget.LAUNCH_VEHICLES[0]
    assert budget.launches_required(63_800.0 * 10, v) == pytest.approx(10)
    assert 3.1e3 < budget.leo_to_escape_dv() < 3.3e3
    assert 0.4 < budget.delivered_fraction("chemical") < 0.55
    assert budget.launches_required(1e9, v, "chemical") > budget.launches_required(1e9, v, "self")
    assert budget.launch_cost(2e10, 50.0) == pytest.approx(1e12)


def test_sensitivity_elasticities():
    s = budget.sensitivity()
    assert s["f"][0] == pytest.approx(1.0, abs=1e-6)       # mass is proportional to the target
    assert 0 < s["sigma"][0] < 1                          # heavier film -> heavier shade, but closer in
    assert s["q"][0] > 0                                  # more radiation pressure -> further out -> more mass
    assert s["u"][0] < 0                                  # limb darkening helps
    assert max(s, key=lambda p: abs(s[p][0])) == "f"
