"""Phase 2 validation: finite-disk shadow optics."""
import numpy as np
import pytest

from src import constants as k
from src.optics import shadow

L1 = 1.5e9   # m


def test_limb_darkened_disk_is_normalised():
    r = np.linspace(0, 1, 200001)
    for u in (0.0, 0.6, 1.0):
        total = np.trapezoid(shadow.limb_darkened_intensity(r, u) * 2 * np.pi * r, r)
        assert total == pytest.approx(1.0, rel=1e-4)


def test_sun_angular_radius():
    assert np.degrees(shadow.solar_angular_radius()) * 2 == pytest.approx(0.533, abs=0.002)


def test_penumbra_at_l1_is_wider_than_the_earth():
    assert shadow.penumbra_radius_at_earth(L1) > k.R_EARTH
    assert shadow.penumbra_radius_at_earth(L1) == pytest.approx(7046e3, rel=1e-3)


def test_no_umbra_reaches_earth():
    radius = np.sqrt(3.4e12 / np.pi)                       # Angel-sized shade as one disk
    assert shadow.umbra_length(radius, L1) < 0.2 * L1      # umbra ends long before Earth
    assert shadow.shade_radius_for_umbra_at_earth(L1) > 6 * radius


def test_uniform_sun_efficiency_has_closed_form():
    """Without limb darkening the on-axis efficiency is (R_E D / (R_sun d))^2."""
    exact = (k.R_EARTH * k.AU / (k.R_SUN * L1)) ** 2
    assert shadow.point_efficiency(0.0, L1, u=0.0) == pytest.approx(exact, rel=2e-3)


def test_energy_is_conserved_by_the_penumbra():
    """Integrated over the whole penumbra (not just Earth) a shade element removes exactly
    the light it intercepts: integral of blocking dA_p = (D/(D-d))^2."""
    d = L1
    r_pen = shadow.penumbra_radius_at_earth(d)
    r = np.linspace(0, r_pen, 20001)
    b = shadow.point_blocking(r, 0.0, 0.0, 0.0, d)
    total = np.trapezoid(b * 2 * np.pi * r, r)
    assert total == pytest.approx((k.AU / (k.AU - d)) ** 2, rel=1e-3)


def test_dimming_is_faint_and_spread_over_the_whole_disk():
    radius = np.sqrt(3.4e12 / np.pi)
    p = np.linspace(0, k.R_EARTH, 30)
    dim = shadow.dimming_profile(p, L1, radius)
    assert dim.max() < 0.04                 # nowhere darker than a few per cent
    assert dim.min() > 0.3 * dim.max()      # every point of the sunlit face is dimmed
    assert shadow.dimming_profile([1.2 * shadow.penumbra_radius_at_earth(L1, radius)], L1, radius)[0] == 0.0


@pytest.mark.parametrize("d, angel_eps", [(1.5e9, 0.68), (3.0e9, 0.25)])
def test_efficiency_matches_angel_2006(d, angel_eps):
    assert shadow.monthly_mean_efficiency(d) == pytest.approx(angel_eps, abs=0.02)


def test_efficiency_falls_with_distance_and_spread():
    d = np.linspace(1.2e9, 4e9, 12)
    eps = [shadow.monthly_mean_efficiency(x) for x in d]
    assert np.all(np.diff(eps) < 0)
    assert shadow.efficiency(L1, 4000e3) < shadow.efficiency(L1, 1000e3) < shadow.efficiency(L1, 0.0)


def test_blocked_fraction_scales_linearly_with_area_for_a_cloud():
    f1 = shadow.blocked_fraction(1e12, L1, solid_disk=False)
    f2 = shadow.blocked_fraction(2e12, L1, solid_disk=False)
    assert f2 == pytest.approx(2 * f1, rel=1e-12)


def test_required_area_roundtrip():
    for f in (0.01, 0.018, 0.02):
        a = shadow.required_area(f, L1)
        assert shadow.blocked_fraction(a, L1) == pytest.approx(f, rel=1e-8)


@pytest.mark.parametrize("d, angel_area_km2", [(1.5e9, 3.4e6), (3.0e9, 9.4e6)])
def test_required_area_same_order_as_angel_2006(d, angel_area_km2):
    """REQUIRED: shade area for a 1.8 % reduction is the same order of magnitude as Angel (2006).

    Angel quotes 3.4 million km^2 at 1.5 Gm and 9.4 million km^2 at 3 Gm.  The independent
    calculation here agrees to a few per cent, far better than an order of magnitude.
    """
    area_km2 = shadow.required_area(0.018, d) / 1e6
    assert 0.1 < area_km2 / angel_area_km2 < 10          # the required criterion
    assert area_km2 == pytest.approx(angel_area_km2, rel=0.05)   # what is actually achieved


def test_solid_angle_estimate_matches_sanchez_mcinnes_2015():
    """Their Eq. 1 quick estimate: 915 km radius, 2.6e6 km^2 for 1.7 % at ~1.5e6 km."""
    area = shadow.solid_angle_area(0.017, L1)
    assert area / 1e12 == pytest.approx(2.6, rel=0.02)
    assert np.sqrt(area / np.pi) / 1e3 == pytest.approx(915, rel=0.01)
