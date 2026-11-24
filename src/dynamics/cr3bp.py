"""Sun-Earth circular restricted three-body problem (CR3BP) with solar radiation pressure.

Frame and units
---------------
Rotating frame centred on the Sun-secondary barycentre, x axis from the Sun to the
secondary, z axis along the orbital angular momentum.  Non-dimensional units:

* length  : Sun-secondary distance (1 AU)
* mass    : m_sun + m_secondary
* time    : 1/n, so one sidereal year is 2*pi time units

The Sun sits at (-mu, 0, 0) and the secondary at (1 - mu, 0, 0).

Solar radiation pressure (SRP)
------------------------------
For a flat shade that always faces the Sun, the radiation force points radially away
from the Sun and falls off as 1/r^2, exactly like solar gravity.  It therefore acts as a
reduction of the Sun's gravity by the factor (1 - beta), where the *lightness number*

    beta = F_rad / F_grav = Q * sigma_star / sigma,
    sigma_star = L_sun / (2 pi G M_sun c)  ~ 1.53 g/m^2

sigma is the shade's mass per unit area and Q is the radiation-pressure efficiency
relative to a perfect mirror (Q = 1 perfect specular reflector, Q = 0.5 perfect
absorber, Q -> 0 for a transparent/refracting screen).  Q is what Angel (2006) calls
the "effective reflectivity R".

Equations of motion (state = [x, y, z, vx, vy, vz]):

    x'' - 2 y' = dU/dx,   y'' + 2 x' = dU/dy,   z'' = dU/dz
    U = (x^2 + y^2)/2 + (1 - beta)(1 - mu)/r1 + mu/r2

and the Jacobi constant  C = 2 U - v^2  is conserved for constant beta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from src import constants as k

RTOL = 1e-12
ATOL = 1e-14


@dataclass(frozen=True)
class System:
    """A Sun + secondary CR3BP system and its unit conversions."""

    name: str
    gm_secondary: float          # m^3/s^2
    gm_sun: float = k.GM_SUN     # m^3/s^2
    length: float = k.AU         # m

    @property
    def mu(self) -> float:
        return self.gm_secondary / (self.gm_sun + self.gm_secondary)

    @property
    def mean_motion(self) -> float:
        """rad/s"""
        return np.sqrt((self.gm_sun + self.gm_secondary) / self.length**3)

    @property
    def time(self) -> float:
        """seconds per non-dimensional time unit"""
        return 1.0 / self.mean_motion

    @property
    def velocity(self) -> float:
        """m/s per non-dimensional velocity unit"""
        return self.length / self.time

    @property
    def accel(self) -> float:
        """m/s^2 per non-dimensional acceleration unit"""
        return self.length / self.time**2

    @property
    def year(self) -> float:
        """One orbital period of the primaries, in non-dimensional time (= 2 pi)."""
        return 2.0 * np.pi


# The standard choice for Sun-Earth libration-point work: the secondary is the
# Earth-Moon system lumped at its barycentre.
SUN_EARTH_MOON = System("Sun - (Earth+Moon)", k.GM_EARTH + k.GM_MOON)
# Earth alone, for comparison.
SUN_EARTH = System("Sun - Earth", k.GM_EARTH)
DEFAULT = SUN_EARTH_MOON


# --------------------------------------------------------------------------------------
# Lightness number
# --------------------------------------------------------------------------------------
def lightness_number(sigma: float, q: float = 1.0) -> float:
    """beta for areal density sigma [kg/m^2] and radiation-pressure efficiency q."""
    return q * k.SIGMA_STAR / sigma


def areal_density_for_beta(beta: float, q: float = 1.0) -> float:
    """Areal density [kg/m^2] giving lightness number beta."""
    return q * k.SIGMA_STAR / beta


# --------------------------------------------------------------------------------------
# Potential, equations of motion, Jacobi constant
# --------------------------------------------------------------------------------------
def _radii(x, y, z, mu):
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    return r1, r2


def effective_potential(pos, mu: float, beta: float = 0.0):
    x, y, z = pos
    r1, r2 = _radii(x, y, z, mu)
    return 0.5 * (x**2 + y**2) + (1 - beta) * (1 - mu) / r1 + mu / r2


def acceleration(state, mu: float, beta: float = 0.0) -> np.ndarray:
    """Rotating-frame acceleration [ax, ay, az] including Coriolis terms."""
    x, y, z, vx, vy, vz = state
    r1, r2 = _radii(x, y, z, mu)
    k1 = (1 - beta) * (1 - mu) / r1**3
    k2 = mu / r2**3
    ax = 2 * vy + x - k1 * (x + mu) - k2 * (x - 1 + mu)
    ay = -2 * vx + y - k1 * y - k2 * y
    az = -k1 * z - k2 * z
    return np.array([ax, ay, az])


def eom(t, state, mu: float, beta: float = 0.0) -> np.ndarray:
    return np.concatenate([state[3:6], acceleration(state, mu, beta)])


def potential_hessian(pos, mu: float, beta: float = 0.0) -> np.ndarray:
    """3x3 matrix of second derivatives of U."""
    x, y, z = pos
    r1, r2 = _radii(x, y, z, mu)
    a = (1 - beta) * (1 - mu)
    d1 = np.array([x + mu, y, z])
    d2 = np.array([x - 1 + mu, y, z])
    h = -(a / r1**3 + mu / r2**3) * np.eye(3)
    h += 3 * a * np.outer(d1, d1) / r1**5 + 3 * mu * np.outer(d2, d2) / r2**5
    h[0, 0] += 1.0
    h[1, 1] += 1.0
    return h


def jacobian(pos, mu: float, beta: float = 0.0) -> np.ndarray:
    """6x6 matrix A of the linearised system d(delta)/dt = A delta."""
    a = np.zeros((6, 6))
    a[0:3, 3:6] = np.eye(3)
    a[3:6, 0:3] = potential_hessian(pos, mu, beta)
    a[3, 4] = 2.0
    a[4, 3] = -2.0
    return a


def eom_stm(t, y, mu: float, beta: float = 0.0) -> np.ndarray:
    """State (6) + state-transition matrix (36) propagated together."""
    state = y[:6]
    phi = y[6:].reshape(6, 6)
    dphi = jacobian(state[:3], mu, beta) @ phi
    return np.concatenate([eom(t, state, mu, beta), dphi.ravel()])


def jacobi_constant(state, mu: float, beta: float = 0.0):
    state = np.asarray(state)
    pos = state[0:3]
    v2 = np.sum(state[3:6] ** 2, axis=0)
    return 2 * effective_potential(pos, mu, beta) - v2


def propagate(state0, t_span, mu: float, beta: float = 0.0, *, t_eval=None, events=None,
              dense_output=False, rtol: float = RTOL, atol: float = ATOL, stm: bool = False):
    """Integrate with DOP853 (8th-order Dormand-Prince), rtol <= 1e-10 as required."""
    assert rtol <= 1e-10
    if stm:
        y0 = np.concatenate([np.asarray(state0, float), np.eye(6).ravel()])
        fun = eom_stm
    else:
        y0 = np.asarray(state0, float)
        fun = eom
    return solve_ivp(fun, t_span, y0, method="DOP853", args=(mu, beta), t_eval=t_eval,
                     events=events, dense_output=dense_output, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------------------
# Collinear equilibrium between the Sun and the Earth (L1, and its SRP-displaced version)
# --------------------------------------------------------------------------------------
def _dudx_on_axis(x: float, mu: float, beta: float) -> float:
    """dU/dx on the x axis between the primaries (-mu < x < 1 - mu)."""
    return x - (1 - beta) * (1 - mu) / (x + mu) ** 2 + mu / (x - 1 + mu) ** 2


def l1_position(mu: float, beta: float = 0.0) -> float:
    """x coordinate of the sunward collinear equilibrium (classical L1 when beta = 0).

    Root of dU/dx = 0 bracketed strictly between the Sun and the secondary, solved
    with Brent's method to machine precision.
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1): for beta >= 1 radiation pressure "
                         "exceeds solar gravity and no equilibrium exists")
    lo = -mu + 1e-9
    hi = 1 - mu - 1e-9
    return brentq(_dudx_on_axis, lo, hi, args=(mu, beta), xtol=1e-16, rtol=1e-15, maxiter=500)


def l1_position_newton(mu: float, beta: float = 0.0, tol: float = 1e-15) -> float:
    """Independent Newton-Raphson solution, started from the Hill-radius estimate."""
    x = 1 - mu - (mu / 3) ** (1 / 3)
    for _ in range(100):
        f = _dudx_on_axis(x, mu, beta)
        df = 1 + 2 * (1 - beta) * (1 - mu) / (x + mu) ** 3 - 2 * mu / (x - 1 + mu) ** 3
        step = f / df
        x -= step
        if abs(step) < tol:
            break
    return x


def l1_distance_from_secondary(system: System = DEFAULT, beta: float = 0.0) -> float:
    """Distance [m] from the secondary (Earth or Earth-Moon barycentre) to the equilibrium."""
    return (1 - system.mu - l1_position(system.mu, beta)) * system.length


def beta_for_distance(distance_m: float, system: System = DEFAULT) -> float:
    """Lightness number needed for equilibrium at a given distance sunward of the secondary.

    Closed form from dU/dx = 0, which is linear in beta.
    """
    mu = system.mu
    x = 1 - mu - distance_m / system.length
    grav_sun = (1 - mu) / (x + mu) ** 2
    return 1 - (x + mu / (x - 1 + mu) ** 2) / grav_sun


# --------------------------------------------------------------------------------------
# Linear stability of the equilibrium
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Linearisation:
    x_eq: float
    a_planar: np.ndarray        # 4x4 in-plane matrix, state [dx, dy, dvx, dvy]
    lam: float                  # unstable eigenvalue (1/time unit)
    omega: float                # in-plane oscillation frequency
    omega_z: float              # out-of-plane oscillation frequency
    unstable_right: np.ndarray  # right eigenvector for +lam
    stable_right: np.ndarray    # right eigenvector for -lam
    unstable_left: np.ndarray   # left eigenvector for +lam, normalised w.u = 1
    centre_vector: np.ndarray   # real initial condition [1, 0, 0, vy] on the linear periodic orbit


def linearise(mu: float, beta: float = 0.0) -> Linearisation:
    x_eq = l1_position(mu, beta)
    a6 = jacobian([x_eq, 0.0, 0.0], mu, beta)
    idx = [0, 1, 3, 4]
    a = a6[np.ix_(idx, idx)]
    w, v = np.linalg.eig(a)
    i_u = int(np.argmax(w.real))
    i_s = int(np.argmin(w.real))
    i_c = int(np.argmax(w.imag))
    lam = float(w[i_u].real)
    omega = float(w[i_c].imag)
    u = v[:, i_u].real
    u = u / np.linalg.norm(u[:2])
    s = v[:, i_s].real
    s = s / np.linalg.norm(s[:2])
    wl, vl = np.linalg.eig(a.T)
    left = vl[:, int(np.argmax(wl.real))].real
    left = left / (left @ u)
    c = v[:, i_c] / v[0, i_c]          # scale so the x component is 1 (real)
    centre = c.real
    omega_z = float(np.sqrt(-a6[5, 2]))
    return Linearisation(x_eq, a, lam, omega, omega_z, u, s, left, centre)


def e_folding_time_days(system: System = DEFAULT, beta: float = 0.0) -> float:
    lin = linearise(system.mu, beta)
    return system.time / lin.lam / k.DAY


# --------------------------------------------------------------------------------------
# Planar Lyapunov orbit by differential correction (single shooting)
# --------------------------------------------------------------------------------------
@dataclass
class PeriodicOrbit:
    state0: np.ndarray       # [x, 0, 0, 0, vy, 0]
    period: float            # non-dimensional
    mu: float
    beta: float
    iterations: int
    residual: float          # |vx| at the half-period crossing
    monodromy: np.ndarray | None = None

    def sample(self, n: int = 400, revolutions: float = 1.0):
        t = np.linspace(0, self.period * revolutions, n)
        sol = propagate(self.state0, (0, t[-1]), self.mu, self.beta, t_eval=t)
        return sol.t, sol.y

    @property
    def stability_multiplier(self) -> float:
        """Largest |eigenvalue| of the monodromy matrix: error growth per revolution."""
        return float(np.max(np.abs(np.linalg.eigvals(self.monodromy))))


def _y_crossing(t, y, mu, beta):
    return y[1]


_y_crossing.terminal = True
_y_crossing.direction = 0


def _half_period(state0, mu, beta, t_guess):
    """Integrate state+STM from the x axis to the next perpendicular-plane crossing."""
    sol = propagate(state0, (0.0, 3.0 * t_guess), mu, beta, stm=True,
                    events=_HalfCrossing(1e-3 * t_guess))
    if sol.t_events[0].size == 0:
        raise RuntimeError("orbit did not return to the x axis")
    return sol.t[-1], sol.y[:6, -1], sol.y[6:, -1].reshape(6, 6)


class _HalfCrossing:
    """Event y = 0, ignored during the first instants after departure from the axis."""

    terminal = True
    direction = 0

    def __init__(self, t_min):
        self.t_min = t_min

    def __call__(self, t, y, mu, beta):
        return y[1] if t > self.t_min else 1.0 * np.sign(y[4] if y[4] != 0 else 1.0)


def lyapunov_orbit(amplitude_x: float, mu: float, beta: float = 0.0, *, tol: float = 1e-12,
                   max_iter: int = 30, guess: tuple[float, float] | None = None) -> PeriodicOrbit:
    """Planar Lyapunov orbit about the (SRP-displaced) L1 point.

    amplitude_x : distance of the starting point from the equilibrium, on its Earth side
                  (non-dimensional).
    Method      : the orbit is symmetric about the x axis, so a trajectory leaving the
                  axis perpendicularly, [x0, 0, 0, vy0], is periodic if it next crosses
                  the axis perpendicularly as well (vx = 0).  x0 is held fixed and vy0 is
                  corrected by Newton iteration using the state-transition matrix:

                      d(vx_f) = (Phi[vx,vy0] - (ax_f / vy_f) * Phi[y,vy0]) d(vy0)

                  The initial guess comes from the linearised centre mode.
    """
    lin = linearise(mu, beta)
    x0 = lin.x_eq + amplitude_x
    if guess is None:
        vy0 = lin.centre_vector[3] * amplitude_x
        t_half_guess = np.pi / lin.omega
    else:
        vy0, t_half_guess = guess
    it = 0
    for it in range(1, max_iter + 1):
        s0 = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        t_half, sf, phi = _half_period(s0, mu, beta, t_half_guess)
        vx_f = sf[3]
        if abs(vx_f) < tol:
            break
        ax_f = acceleration(sf, mu, beta)[0]
        dvx_dvy0 = phi[3, 4] - (ax_f / sf[4]) * phi[1, 4]
        vy0 -= vx_f / dvx_dvy0
        t_half_guess = t_half
    s0 = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
    period = 2 * t_half
    sol = propagate(s0, (0.0, period), mu, beta, stm=True)
    mono = sol.y[6:, -1].reshape(6, 6)
    return PeriodicOrbit(s0, period, mu, beta, it, abs(vx_f), mono)


def lyapunov_family(amplitudes, mu: float, beta: float = 0.0) -> list[PeriodicOrbit]:
    """Continue the family in amplitude, each orbit seeding the next (natural continuation)."""
    orbits: list[PeriodicOrbit] = []
    guess = None
    prev_amp = None
    for amp in amplitudes:
        if guess is not None:
            guess = (guess[0] * amp / prev_amp, guess[1])
        orb = lyapunov_orbit(amp, mu, beta, guess=guess)
        orbits.append(orb)
        guess = (orb.state0[4], orb.period / 2)
        prev_amp = amp
    return orbits


# --------------------------------------------------------------------------------------
# Instability demonstration
# --------------------------------------------------------------------------------------
def divergence_from_equilibrium(offset_m: float, system: System = DEFAULT, beta: float = 0.0,
                                days: float = 365.25, n: int = 2000, direction=(1.0, 0.0)):
    """Release a shade at rest (rotating frame) offset_m from the equilibrium.

    Returns (t_days, separation_m, states).  Integration stops if the shade leaves the
    L1 neighbourhood (0.02 AU from the equilibrium) since it is then simply lost.
    """
    mu = system.mu
    x_eq = l1_position(mu, beta)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d) * offset_m / system.length
    s0 = np.array([x_eq + d[0], d[1], 0.0, 0.0, 0.0, 0.0])
    t_end = days * k.DAY / system.time

    def lost(t, y, mu, beta):
        return np.hypot(y[0] - x_eq, y[1]) - 0.02

    lost.terminal = True
    sol = propagate(s0, (0, t_end), mu, beta, t_eval=np.linspace(0, t_end, n), events=lost)
    sep = np.sqrt((sol.y[0] - x_eq) ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2) * system.length
    return sol.t * system.time / k.DAY, sep, sol.y
