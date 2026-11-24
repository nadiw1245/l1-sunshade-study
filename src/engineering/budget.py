"""Mass, launch and cost budget of an L1 sunshade, tied to the physics modules.

The chain of reasoning for one design point is

    areal density sigma, radiation-pressure efficiency Q
        -> lightness number beta                      (src.dynamics.cr3bp)
        -> equilibrium distance d from Earth          (src.dynamics.cr3bp)
        -> blocking efficiency eps(d)                 (src.optics.shadow)
        -> area A = f pi R_E^2 / eps                  (src.optics.shadow)
        -> mass M = A sigma
        -> launches = M / (payload delivered per launch)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import constants as k
from src.dynamics import cr3bp
from src.optics import shadow

G0 = 9.80665


@dataclass(frozen=True)
class LaunchVehicle:
    name: str
    payload_leo_kg: float
    note: str


# Payload-to-LEO figures as published by the operators (see docs/background.md).
LAUNCH_VEHICLES = (
    LaunchVehicle("Falcon Heavy (expendable)", 63_800.0, "SpaceX published capability"),
    LaunchVehicle("SLS Block 1", 95_000.0, "NASA published capability"),
    LaunchVehicle("Starship (reusable)", 100_000.0, "SpaceX design target ('100+ t'), not yet demonstrated"),
)


@dataclass
class Design:
    f: float                 # insolation reduction
    sigma: float             # kg/m^2
    q: float                 # radiation-pressure efficiency
    beta: float
    distance: float          # m, equilibrium distance from the Earth-Moon barycentre
    efficiency: float
    area: float              # m^2
    mass: float              # kg

    @property
    def disk_radius(self) -> float:
        return float(np.sqrt(self.area / np.pi))


def design(f: float, sigma: float, q: float = 1.0, *, monthly: bool = True,
           u: float = shadow.LIMB_DARKENING_U, system: cr3bp.System = cr3bp.DEFAULT,
           spread_radius: float | None = None) -> Design:
    """Self-consistent shade: it sits where gravity, rotation and radiation pressure balance.

    spread_radius=None models one solid disk; a number models a cloud of flyers spread
    uniformly over a disk of that radius.
    """
    beta = cr3bp.lightness_number(sigma, q)
    d = cr3bp.l1_distance_from_secondary(system, beta)
    if spread_radius is None:
        area = shadow.required_area(f, d, monthly=monthly, u=u)
    else:
        area = shadow.required_area(f, d, solid_disk=False, spread_radius=spread_radius,
                                    monthly=monthly, u=u)
    eps = f * np.pi * k.R_EARTH**2 / area
    return Design(f, sigma, q, beta, d, eps, area, area * sigma)


def shade_mass(area: float, sigma: float) -> float:
    return area * sigma


def leo_to_escape_dv(altitude: float = 200e3) -> float:
    """Delta-v from a circular low Earth orbit to Earth escape (C3 = 0): (sqrt2 - 1) v_circ.

    L1 lies at the edge of Earth's gravitational reach, so this is a good first estimate
    of the impulsive cost of leaving LEO for L1.
    """
    v_circ = np.sqrt(k.GM_EARTH / (k.R_EARTH + altitude))
    return float((np.sqrt(2) - 1) * v_circ)


def delivered_fraction(transfer: str = "self", isp: float = 450.0, altitude: float = 200e3) -> float:
    """Fraction of the mass placed in LEO that arrives at L1 as shade.

    "self"     : the shade elements sail/ion-propel themselves from LEO (Angel 2006;
                 Fuglesang & Garcia de Herreros 2021) -> 1.0 (optimistic upper bound)
    "chemical" : an impulsive chemical stage of specific impulse `isp` provides the
                 escape delta-v; ideal rocket equation, stage dry mass neglected.
    """
    if transfer == "self":
        return 1.0
    if transfer == "chemical":
        return float(np.exp(-leo_to_escape_dv(altitude) / (isp * G0)))
    raise ValueError(transfer)


def launches_required(mass: float, vehicle: LaunchVehicle, transfer: str = "self",
                      packing: float = 1.0) -> float:
    """Number of launches; `packing` is the fraction of payload mass that is shade."""
    return mass / (vehicle.payload_leo_kg * packing * delivered_fraction(transfer))


def launch_cost(mass: float, cost_per_kg: float, transfer: str = "self") -> float:
    """Launch cost [USD] at a given price per kg to LEO."""
    return mass / delivered_fraction(transfer) * cost_per_kg


def launches_per_day(n_launches: float, years: float) -> float:
    return n_launches / (years * 365.25)


# --------------------------------------------------------------------------------------
# Sensitivity analysis
# --------------------------------------------------------------------------------------
BASELINE = dict(f=0.018, sigma=5.6e-3, q=0.10, u=shadow.LIMB_DARKENING_U,
                wobble=k.EARTH_BARYCENTRE_OFFSET)


def total_mass(f, sigma, q, u, wobble, system: cr3bp.System = cr3bp.DEFAULT) -> float:
    """Shade mass [kg] as a function of the main free parameters (cloud of small flyers
    close to the axis, i.e. the point-element efficiency)."""
    beta = cr3bp.lightness_number(sigma, q)
    d = cr3bp.l1_distance_from_secondary(system, beta)
    eps = shadow.monthly_mean_efficiency(d, 0.0, wobble=wobble, u=u)
    return f * np.pi * k.R_EARTH**2 / eps * sigma


def sensitivity(func=total_mass, baseline: dict | None = None, rel_step: float = 0.10) -> dict:
    """One-at-a-time sensitivity.

    Returns {parameter: (elasticity, value_low, value_high)} where
    elasticity = d ln(output) / d ln(parameter) from a central difference of +-rel_step,
    and value_low/high are the outputs with the parameter scaled by (1 -/+ rel_step).
    """
    baseline = dict(BASELINE if baseline is None else baseline)
    out = {}
    for name, value in baseline.items():
        lo = func(**{**baseline, name: value * (1 - rel_step)})
        hi = func(**{**baseline, name: value * (1 + rel_step)})
        elasticity = (np.log(hi) - np.log(lo)) / (np.log(1 + rel_step) - np.log(1 - rel_step))
        out[name] = (float(elasticity), float(lo), float(hi))
    return out
