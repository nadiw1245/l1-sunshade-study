"""Physical constants (SI) used throughout the study.

Values are CODATA / IAU nominal values; each is a measured constant, not a tuned parameter.
"""
import math

G = 6.67430e-11                # m^3 kg^-1 s^-2 (CODATA 2018)
C_LIGHT = 299_792_458.0        # m/s (exact)
SIGMA_SB = 5.670374419e-8      # W m^-2 K^-4 (CODATA 2018)
AU = 1.495978707e11            # m (IAU 2012, exact)
GM_SUN = 1.32712440018e20      # m^3/s^2
GM_EARTH = 3.986004418e14      # m^3/s^2
GM_MOON = 4.9048695e12         # m^3/s^2
L_SUN = 3.828e26               # W (IAU 2015 nominal)
R_SUN = 6.957e8                # m (IAU 2015 nominal)
R_EARTH = 6.371e6              # m (mean radius)
S0 = 1361.0                    # W/m^2 total solar irradiance at 1 AU (Kopp & Lean 2011)
EARTH_MOON_DISTANCE = 3.844e8  # m (mean)
YEAR = 365.25 * 86400.0        # s
DAY = 86400.0

M_SUN = GM_SUN / G
# Distance of Earth's centre from the Earth-Moon barycentre (~4671 km)
EARTH_BARYCENTRE_OFFSET = EARTH_MOON_DISTANCE * GM_MOON / (GM_EARTH + GM_MOON)

# Critical areal density: a perfectly reflecting flat sail facing the Sun with this
# mass per area feels radiation pressure equal to solar gravity (beta = 1).
# sigma* = L_sun / (2 pi G M_sun c)  ~ 1.53 g/m^2
SIGMA_STAR = L_SUN / (2.0 * math.pi * GM_SUN * C_LIGHT)  # kg/m^2
