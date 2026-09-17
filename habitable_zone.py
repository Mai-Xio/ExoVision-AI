"""
Habitable-zone boundaries.

Implements the Kopparapu et al. (2013, ApJ 765, 131; 2014 erratum) parametrisation
of the stellar-flux limits of the circumstellar habitable zone. The effective
stellar flux at each boundary is a quartic in the host's effective temperature:

    S_eff = S_eff_sun + a*T + b*T^2 + c*T^3 + d*T^4,    T = Teff - 5780 K

and the corresponding orbital distance is

    d [au] = sqrt( (L/L_sun) / S_eff )

Four boundaries, two bracketing pairs:

  CONSERVATIVE   runaway greenhouse  ->  maximum greenhouse
                 Theoretical limits for a cloud-free CO2/H2O atmosphere on a
                 rocky planet. These coefficients give 0.98 - 1.69 au for the Sun.

  OPTIMISTIC     recent Venus  ->  early Mars
                 Empirical limits from Solar System history: Venus appears to
                 have lost its water ~1 Gyr ago and Mars appears to have had
                 surface water ~3.8 Gyr ago. These coefficients give
                 0.75 - 1.77 au for the Sun.

A note on which numbers you will see elsewhere
----------------------------------------------
Published solar inner edges range from about 0.95 to 0.99 au depending on the
revision and on whether the RUNAWAY or the MOIST greenhouse limit is used. The
coefficients below are the runaway-greenhouse set and yield 0.98 au; the widely
quoted 0.95 au comes from a later revision of that same boundary. The spread is
roughly 4%, which is smaller than the stellar-parameter uncertainty on most KOIs
(see docs/02-eda-findings.md), so it does not drive the result here. It is
recorded because silently matching a number to the wrong revision is how
constants rot. `hz_reference_values()` returns what this implementation actually
produces, and the tests assert against that rather than against a remembered
literature figure.

The two are reported separately throughout. A candidate inside the optimistic
zone and outside the conservative one is a materially different claim from one
inside both, and collapsing them into a single "in the HZ" boolean destroys the
distinction that the literature exists to make.

What this does NOT do
---------------------
The habitable zone is a *liquid-water proxy* under stated atmospheric
assumptions. It says nothing about whether a planet has an atmosphere, retained
its water, has a magnetic field, is tidally locked, or experiences the flare
activity typical of M dwarfs. Being in the zone is necessary-ish, not
sufficient, and not evidence of habitability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

#: Reference temperature of the parametrisation.
TEFF_REF_K: Final[float] = 5780.0

#: Validity range stated by Kopparapu et al. Outside it the quartic is an
#: extrapolation and results must be flagged.
TEFF_VALID_MIN_K: Final[float] = 2600.0
TEFF_VALID_MAX_K: Final[float] = 7200.0


@dataclass(frozen=True)
class HZCoefficients:
    """Quartic coefficients for one habitable-zone boundary."""

    name: str
    s_eff_sun: float
    a: float
    b: float
    c: float
    d: float

    def s_eff(self, teff_k: np.ndarray | float) -> np.ndarray:
        t = np.asarray(teff_k, dtype=float) - TEFF_REF_K
        return self.s_eff_sun + self.a * t + self.b * t**2 + self.c * t**3 + self.d * t**4


# Kopparapu et al. (2013) Table 3, with the 2014 erratum values.
RECENT_VENUS = HZCoefficients("recent_venus", 1.7763, 1.4335e-4, 3.3954e-9,
                              -7.6364e-12, -1.1950e-15)
RUNAWAY_GREENHOUSE = HZCoefficients("runaway_greenhouse", 1.0385, 1.2456e-4, 1.4612e-8,
                                    -7.6345e-12, -1.7511e-15)
MAXIMUM_GREENHOUSE = HZCoefficients("maximum_greenhouse", 0.3507, 5.9578e-5, 1.6707e-9,
                                    -3.0058e-12, -5.1925e-16)
EARLY_MARS = HZCoefficients("early_mars", 0.3207, 5.4471e-5, 1.5275e-9,
                            -2.1709e-12, -3.8282e-16)

CONSERVATIVE_BOUNDS: Final[tuple[HZCoefficients, HZCoefficients]] = (
    RUNAWAY_GREENHOUSE, MAXIMUM_GREENHOUSE)
OPTIMISTIC_BOUNDS: Final[tuple[HZCoefficients, HZCoefficients]] = (
    RECENT_VENUS, EARLY_MARS)


@dataclass(frozen=True)
class HabitableZone:
    """Flux and distance limits for one host star.

    Flux limits are ordered inner (higher flux) to outer (lower flux).
    """

    teff_k: float
    luminosity_sun: float
    s_inner_conservative: float
    s_outer_conservative: float
    s_inner_optimistic: float
    s_outer_optimistic: float
    extrapolated: bool

    @property
    def distance_conservative_au(self) -> tuple[float, float]:
        return (float(np.sqrt(self.luminosity_sun / self.s_inner_conservative)),
                float(np.sqrt(self.luminosity_sun / self.s_outer_conservative)))

    @property
    def distance_optimistic_au(self) -> tuple[float, float]:
        return (float(np.sqrt(self.luminosity_sun / self.s_inner_optimistic)),
                float(np.sqrt(self.luminosity_sun / self.s_outer_optimistic)))

    def contains_flux(self, insolation_earth: float, optimistic: bool = False) -> bool:
        lo, hi = ((self.s_outer_optimistic, self.s_inner_optimistic) if optimistic
                  else (self.s_outer_conservative, self.s_inner_conservative))
        return bool(lo <= insolation_earth <= hi)


def habitable_zone(teff_k: float, luminosity_sun: float) -> HabitableZone:
    """Compute both bracketing pairs for a host star."""
    extrapolated = not (TEFF_VALID_MIN_K <= teff_k <= TEFF_VALID_MAX_K)
    return HabitableZone(
        teff_k=float(teff_k),
        luminosity_sun=float(luminosity_sun),
        s_inner_conservative=float(RUNAWAY_GREENHOUSE.s_eff(teff_k)),
        s_outer_conservative=float(MAXIMUM_GREENHOUSE.s_eff(teff_k)),
        s_inner_optimistic=float(RECENT_VENUS.s_eff(teff_k)),
        s_outer_optimistic=float(EARLY_MARS.s_eff(teff_k)),
        extrapolated=extrapolated,
    )


def hz_flux_bounds(
    teff_k: np.ndarray | float, optimistic: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised (outer_flux, inner_flux) limits in Earth flux units."""
    inner_c, outer_c = (OPTIMISTIC_BOUNDS if optimistic else CONSERVATIVE_BOUNDS)
    return np.asarray(outer_c.s_eff(teff_k)), np.asarray(inner_c.s_eff(teff_k))


def hz_position(
    insolation_earth: np.ndarray | float,
    teff_k: np.ndarray | float,
    optimistic: bool = False,
) -> np.ndarray:
    """
    Fractional position across the habitable zone, in log-flux space.

    0 = inner edge, 1 = outer edge, values outside [0, 1] are outside the zone.
    Log space because flux falls as 1/d^2, so equal linear steps in flux are
    very unequal steps in orbital distance.
    """
    outer, inner = hz_flux_bounds(teff_k, optimistic)
    s = np.asarray(insolation_earth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log10(inner) - np.log10(s)) / (np.log10(inner) - np.log10(outer))


def hz_reference_values() -> dict[str, tuple[float, float]]:
    """Solar habitable-zone limits as produced by THIS implementation.

    Regression anchor for the tests: if a coefficient is ever edited, these
    change and the test fails loudly rather than shifting every score quietly.
    """
    sun = habitable_zone(TEFF_REF_K, 1.0)
    return {
        "conservative_au": sun.distance_conservative_au,
        "optimistic_au": sun.distance_optimistic_au,
        "conservative_flux": (sun.s_outer_conservative, sun.s_inner_conservative),
        "optimistic_flux": (sun.s_outer_optimistic, sun.s_inner_optimistic),
    }
