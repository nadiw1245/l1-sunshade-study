"""Shadow optics of a sunshade between the Sun and the Earth.

Geometry (small-angle, all angles << 1 rad)
-------------------------------------------
Earth's centre is the origin, the Sun's centre is on the axis at distance D, and the
shade lies in a plane at distance d from Earth.  An observer at transverse position p
on Earth looking at a shade element at transverse position q sees it at an angle

    psi = | q/d - p (1/d - 1/D) |

from the centre of the solar disk, whose angular radius is theta_s = R_sun / D.

The Sun is a finite, limb-darkened disk with the linear law

    I(psi) = I0 [1 - u (1 - m)],   m = sqrt(1 - (psi/theta_s)^2),   psi <= theta_s

whose disk-integrated flux is F = pi theta_s^2 I0 (1 - u/3).  An opaque element of
area dA removes the fraction  I(psi) dA / (d^2 F)  of the sunlight reaching the observer.

Key quantities
--------------
blocking efficiency  eps = (sunlight removed from the Earth) / (sunlight intercepted
                     by the shade) -- Angel (2006)'s epsilon.
insolation reduction f = eps * A / (pi R_E^2).

Assumptions: opaque (or fully deflecting) shade elements that do not overlap each other
along the line of sight; shade plane perpendicular to the Sun line; the +-R_E difference
in distance of points on the globe is neglected (0.4 % of d).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from src import constants as k

LIMB_DARKENING_U = 0.6   # linear coefficient, approximate bolometric/visible value for the Sun
D_SUN = k.AU


# --------------------------------------------------------------------------------------
# Solar disk
# --------------------------------------------------------------------------------------
def solar_angular_radius(d_sun: float = D_SUN) -> float:
    return k.R_SUN / d_sun


def limb_darkened_intensity(x, u: float = LIMB_DARKENING_U):
    """Intensity normalised so that its integral over the unit disk (x = psi/theta_s) is 1."""
    x = np.asarray(x, float)
    inside = x < 1.0
    m = np.sqrt(np.clip(1.0 - x**2, 0.0, None))
    return np.where(inside, (1 - u * (1 - m)) / (np.pi * (1 - u / 3)), 0.0)


# --------------------------------------------------------------------------------------
# Characteristic lengths
# --------------------------------------------------------------------------------------
def penumbra_radius_at_earth(d: float, shade_radius: float = 0.0, d_sun: float = D_SUN) -> float:
    """Outer radius at Earth's distance of the penumbra cast by a disk shade."""
    return shade_radius + (k.R_SUN + shade_radius) * d / (d_sun - d)


def umbra_length(shade_radius: float, d: float, d_sun: float = D_SUN) -> float:
    """Length of the full-shadow (umbra) cone behind a disk shade."""
    return shade_radius * (d_sun - d) / (k.R_SUN - shade_radius)


def shade_radius_for_umbra_at_earth(d: float, d_sun: float = D_SUN) -> float:
    """Smallest disk radius whose umbra would just reach Earth (total eclipse on axis)."""
    return k.R_SUN * d / d_sun


# --------------------------------------------------------------------------------------
# Quadrature helpers
# --------------------------------------------------------------------------------------
def _disk_nodes(radius: float, n_r: int = 64, n_phi: int = 128):
    """Nodes/weights for integrals over a disk (Gauss-Legendre in r, uniform in angle)."""
    xr, wr = np.polynomial.legendre.leggauss(n_r)
    r = 0.5 * radius * (xr + 1)
    w_r = 0.5 * radius * wr * r
    phi = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    ww = np.repeat(w_r[:, None], n_phi, axis=1) * (2 * np.pi / n_phi)
    return (rr * np.cos(pp)).ravel(), (rr * np.sin(pp)).ravel(), ww.ravel()


_EARTH_NODES = _disk_nodes(k.R_EARTH)


# --------------------------------------------------------------------------------------
# Point (small) shade element
# --------------------------------------------------------------------------------------
def point_blocking(p_x, p_y, q_x: float, q_y: float, d: float, *, u: float = LIMB_DARKENING_U,
                   d_sun: float = D_SUN):
    """Fraction of sunlight removed at Earth position p per unit shade area at q [1/m^2]."""
    theta_s = solar_angular_radius(d_sun)
    kappa = 1.0 / d - 1.0 / d_sun
    psi = np.hypot(q_x / d - p_x * kappa, q_y / d - p_y * kappa)
    return limb_darkened_intensity(psi / theta_s, u) / (theta_s**2 * d**2)


@lru_cache(maxsize=256)
def _efficiency_curve(d: float, u: float, d_sun: float, n: int = 241):
    """Tabulate the point-element efficiency vs off-axis distance (zero beyond the penumbra)."""
    px, py, w = _EARTH_NODES
    s_max = penumbra_radius_at_earth(d, 0.0, d_sun) + k.R_EARTH
    s_grid = np.linspace(0.0, s_max, n)
    eps = np.array([np.sum(w * point_blocking(px, py, s, 0.0, d, u=u, d_sun=d_sun)) for s in s_grid])
    return s_grid, eps


def point_efficiency(offset, d: float, *, u: float = LIMB_DARKENING_U, d_sun: float = D_SUN):
    """Blocking efficiency of a small shade element `offset` metres off the Sun-Earth axis.

    Defined as in Angel (2006): eps = f * pi R_E^2 / A, i.e. relative to the solar
    constant at Earth.
    """
    s_grid, eps = _efficiency_curve(float(d), float(u), float(d_sun))
    out = np.interp(np.abs(np.asarray(offset, float)), s_grid, eps, right=0.0)
    return out if np.ndim(offset) else float(out)


# --------------------------------------------------------------------------------------
# Extended shades: solid disk, or a cloud of flyers spread over a disk
# --------------------------------------------------------------------------------------
def efficiency(d: float, spread_radius: float = 0.0, *, earth_offset: float = 0.0,
               u: float = LIMB_DARKENING_U, d_sun: float = D_SUN) -> float:
    """Blocking efficiency of shade area spread uniformly over a disk of `spread_radius`.

    Because blocking is additive for non-overlapping elements, the same expression
    describes a single solid disk (spread_radius = its radius) and a dilute cloud of
    small flyers filling a disk of that radius.
    earth_offset is the transverse distance of Earth's centre from the shade axis.
    """
    if spread_radius <= 0:
        return point_efficiency(abs(earth_offset), d, u=u, d_sun=d_sun)
    qx, qy, wq = _disk_nodes(spread_radius, 24, 48)
    s = np.hypot(qx - earth_offset, qy)
    return float(np.sum(wq * point_efficiency(s, d, u=u, d_sun=d_sun)) / (np.pi * spread_radius**2))


def monthly_mean_efficiency(d: float, spread_radius: float = 0.0, *,
                            wobble: float = k.EARTH_BARYCENTRE_OFFSET, n_phase: int = 24,
                            u: float = LIMB_DARKENING_U, d_sun: float = D_SUN) -> float:
    """Efficiency averaged over the Earth's monthly motion about the Earth-Moon barycentre.

    The shade sits on the Sun-barycentre line; Earth's centre is displaced transversely
    by wobble * sin(phase) (the Moon's orbit lies close to the ecliptic).
    """
    phases = (np.arange(n_phase) + 0.5) * 2 * np.pi / n_phase
    return float(np.mean([efficiency(d, spread_radius, earth_offset=wobble * np.sin(ph), u=u,
                                     d_sun=d_sun) for ph in phases]))


def cloud_efficiency(d_centre: float, radius: float, length: float, *, monthly: bool = True,
                     n_slices: int = 9, u: float = LIMB_DARKENING_U) -> float:
    """Efficiency of a cylindrical cloud of flyers: transverse `radius`, axial `length`."""
    xs, ws = np.polynomial.legendre.leggauss(n_slices)
    ds = d_centre + 0.5 * length * xs
    fn = monthly_mean_efficiency if monthly else efficiency
    return float(np.sum(0.5 * ws * np.array([fn(di, radius, u=u) for di in ds])))


def blocked_fraction(area: float, d: float, *, solid_disk: bool = True, spread_radius: float = 0.0,
                     monthly: bool = True, u: float = LIMB_DARKENING_U) -> float:
    """Fractional reduction of Earth's insolation, f = eps * A / (pi R_E^2)."""
    radius = np.sqrt(area / np.pi) if solid_disk else spread_radius
    fn = monthly_mean_efficiency if monthly else efficiency
    return fn(d, radius, u=u) * area / (np.pi * k.R_EARTH**2)


def required_area(f: float, d: float, *, solid_disk: bool = True, spread_radius: float = 0.0,
                  monthly: bool = True, u: float = LIMB_DARKENING_U) -> float:
    """Shade area [m^2] needed to reduce insolation by the fraction f at distance d."""
    if not solid_disk:
        fn = monthly_mean_efficiency if monthly else efficiency
        return f * np.pi * k.R_EARTH**2 / fn(d, spread_radius, u=u)
    a_guess = f * np.pi * k.R_EARTH**2
    return brentq(lambda a: blocked_fraction(a, d, monthly=monthly, u=u) - f,
                  0.2 * a_guess, 200 * a_guess, rtol=1e-10)


def solid_angle_area(f: float, d: float, d_sun: float = D_SUN) -> float:
    """Naive estimate: shade covers the fraction f of the solar disk seen from Earth's centre.

    This is the "quick estimate" of Sanchez & McInnes (2015, Eq. 1); it ignores limb
    darkening and the part of the penumbra that misses the Earth.
    """
    return np.pi * f * (k.R_SUN * d / d_sun) ** 2


# --------------------------------------------------------------------------------------
# Dimming map on the Earth
# --------------------------------------------------------------------------------------
def dimming_profile(p, d: float, shade_radius: float, *, u: float = LIMB_DARKENING_U,
                    d_sun: float = D_SUN) -> np.ndarray:
    """Fraction of sunlight removed vs transverse distance p from the shadow axis,
    for a solid disk shade of radius shade_radius centred on the Sun-Earth line."""
    p = np.atleast_1d(np.asarray(p, float))
    qx, qy, wq = _disk_nodes(shade_radius, 48, 96)
    return np.array([np.sum(wq * point_blocking(pi, 0.0, qx, qy, d, u=u, d_sun=d_sun)) for pi in p])
