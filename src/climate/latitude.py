"""One-dimensional (latitude) diffusive energy balance model, Budyko-Sellers type.

Steady state for the zonal-mean surface temperature T(x), x = sin(latitude):

    0 = (S/4) (1 - f) s(x) (1 - alpha(x)) - eps sigma_SB T^4 + F_CO2
        + d/dx [ D (1 - x^2) dT/dx ]

* s(x)      annual-mean distribution of sunlight (global mean 1).  It is *computed* from
            orbital geometry (circular orbit, obliquity 23.44 deg), not fitted.
* alpha(x)  prescribed albedo alpha0 + alpha2 P2(x): brighter towards the poles.  alpha0 is
            set so that the planetary albedo equals the 0-D value.  The albedo is held
            fixed, so there is no ice-albedo feedback.
* D         meridional heat-transport coefficient [W/m^2/K], a tuned parameter chosen so
            that the equator-to-pole temperature contrast is a few tens of kelvin.
* eps       calibrated so the area-mean temperature is the present-day 288 K.

Purpose: CO2 forcing is (in this model) the same at every latitude, whereas a shade
removes a fixed *fraction* of the local sunlight and therefore acts most strongly in the
tropics.  The two cannot cancel at every latitude at once.

This is an illustration of the mechanism only: no clouds, no circulation changes, no
seasons, no hydrological cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

from src import constants as k
from src.climate.ebm import ALBEDO, CO2_PRESENT, T_PRESENT, co2_forcing

OBLIQUITY_DEG = 23.44


def annual_insolation_distribution(x, obliquity_deg: float = OBLIQUITY_DEG, n: int = 720):
    """s(x): annual-mean insolation at sin(latitude) = x divided by the global mean S/4.

    Daily-mean insolation for solar declination dec at latitude lat:
        Q = (S/pi) (h0 sin(lat) sin(dec) + cos(lat) cos(dec) sin(h0)),  cos(h0) = -tan(lat) tan(dec)
    averaged over a circular orbit (sin(dec) = sin(obliquity) sin(orbital longitude)).
    """
    x = np.atleast_1d(np.asarray(x, float))
    lat = np.arcsin(np.clip(x, -1, 1))[:, None]
    lon = (np.arange(n) + 0.5) * 2 * np.pi / n
    dec = np.arcsin(np.sin(np.radians(obliquity_deg)) * np.sin(lon))[None, :]
    cos_h0 = np.clip(-np.tan(lat) * np.tan(dec), -1.0, 1.0)
    h0 = np.arccos(cos_h0)
    q = (h0 * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(h0)) / np.pi
    return 4.0 * q.mean(axis=1)


def p2(x):
    return 0.5 * (3 * np.asarray(x, float) ** 2 - 1)


@dataclass(frozen=True)
class LatitudeModel:
    n: int = 90
    s0: float = k.S0
    planetary_albedo: float = ALBEDO
    albedo_p2: float = 0.20       # pole-equator albedo contrast parameter (assumption)
    diffusivity: float = 0.55     # W/m^2/K (tuned)
    emissivity: float = 0.61
    co2_ref: float = CO2_PRESENT

    @property
    def x(self) -> np.ndarray:
        return -1 + (np.arange(self.n) + 0.5) * 2 / self.n

    @property
    def latitude_deg(self) -> np.ndarray:
        return np.degrees(np.arcsin(self.x))

    @property
    def insolation(self) -> np.ndarray:
        s = annual_insolation_distribution(self.x)
        return s / s.mean()          # remove the tiny quadrature error in the global mean

    @property
    def albedo(self) -> np.ndarray:
        s = self.insolation
        shape = p2(self.x)
        alpha0 = self.planetary_albedo - self.albedo_p2 * np.mean(s * shape)
        return alpha0 + self.albedo_p2 * shape

    def _diffusion_matrix(self) -> np.ndarray:
        n, dx = self.n, 2 / self.n
        edges = -1 + np.arange(1, n) * dx
        g = self.diffusivity * (1 - edges**2) / dx**2
        m = np.zeros((n, n))
        i = np.arange(n - 1)
        m[i, i] -= g
        m[i, i + 1] += g
        m[i + 1, i + 1] -= g
        m[i + 1, i] += g
        return m

    def solve(self, f: float = 0.0, co2: float | None = None) -> np.ndarray:
        """Steady-state T(x) [K] by Newton iteration."""
        co2 = self.co2_ref if co2 is None else co2
        src = self.s0 / 4 * (1 - f) * self.insolation * (1 - self.albedo) + co2_forcing(co2, self.co2_ref)
        m = self._diffusion_matrix()
        es = self.emissivity * k.SIGMA_SB
        t = np.full(self.n, 288.0)
        for _ in range(60):
            res = src - es * t**4 + m @ t
            step = np.linalg.solve(m - np.diag(4 * es * t**3), -res)
            t = t + step
            if np.max(np.abs(step)) < 1e-11:
                break
        return t

    def global_mean(self, t) -> float:
        return float(np.mean(t))     # cells have equal area in x = sin(latitude)

    def shade_fraction_restoring_global_mean(self, co2: float) -> float:
        target = self.global_mean(self.solve())
        return brentq(lambda f: self.global_mean(self.solve(f, co2)) - target, 0.0, 0.1, xtol=1e-12)


def calibrated_latitude_model(t_mean: float = T_PRESENT, **kwargs) -> LatitudeModel:
    base = LatitudeModel(**kwargs)
    eps = brentq(lambda e: replace(base, emissivity=e).global_mean(replace(base, emissivity=e).solve()) - t_mean,
                 0.4, 0.9, xtol=1e-12)
    return replace(base, emissivity=eps)
