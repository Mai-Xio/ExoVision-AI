"""
EarthLikeScore: a project-authored heuristic, 0-100.

    This score is NOT an official NASA classification, NOT a probability of
    habitability, and NOT evidence that any object supports life. It ranks
    candidates for follow-up attention using published catalogue quantities and
    stated assumptions.

Why no hard cutoffs
-------------------
The obvious design is `if 0.8 <= radius <= 1.25: earth_sized = True`. That
reports precision the data does not contain. From Step 2: planet radius is
`koi_ror x koi_srad`, the transit fits only the ratio, and for ~80% of KOIs the
stellar radius comes from broadband photometry with a median fractional
uncertainty near 30%. A nominal 1.0 R_earth planet is then consistent with
0.7-1.3 R_earth at 1 sigma — wider than the cutoff window itself.

Two consequences, both implemented here:

1. Every component uses a SMOOTH function of its input, so a candidate at
   1.26 R_earth is not categorically different from one at 1.24.
2. The score is propagated through MONTE CARLO sampling of the measurement
   uncertainties, and reported as an interval. A score of "72" is meaningless;
   "72, 5th-95th percentile 51-86" is a usable statement.

Component weights are configuration, not code (`ScoringConfig`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ml.preprocessing import constants as C
from ml.earthlike.habitable_zone import (
    TEFF_VALID_MAX_K,
    TEFF_VALID_MIN_K,
    habitable_zone,
    hz_flux_bounds,
)

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "EarthLikeScore is a project-specific heuristic, not an official NASA "
    "habitability classification. It does not establish that a planet is "
    "habitable or that it supports life."
)

SCORE_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 30, "Low similarity to Earth-like criteria"),
    (31, 60, "Possible candidate"),
    (61, 80, "Strong candidate"),
    (81, 100, "High-priority candidate"),
)


def band_for(score: float) -> str:
    for lo, hi, label in SCORE_BANDS:
        if lo <= score <= hi:
            return label
    return "Unscored"


@dataclass(frozen=True)
class ScoringConfig:
    """Weights and tolerances. All configurable; none are physical constants."""

    w_radius: float = 0.30
    w_flux: float = 0.25
    w_temperature: float = 0.15
    w_habitable_zone: float = 0.20
    w_host_star: float = 0.10

    #: Log-space width of the radius kernel. 0.13 dex ~ a factor of 1.35, so a
    #: 1.35 R_earth planet scores ~0.61 rather than falling off a cliff.
    radius_log_sigma: float = 0.13
    #: Above this radius a planet is very unlikely to be rocky (radius valley;
    #: Fulton et al. 2017). Applied as a soft roll-off, not a cut.
    rocky_ceiling_re: float = 1.8

    #: Log-space width of the insolation kernel, centred on Earth's flux.
    flux_log_sigma: float = 0.30

    #: Equilibrium-temperature window. Centre near Earth's 255 K equilibrium
    #: value, NOT its 288 K surface temperature — koi_teq is a blackbody
    #: quantity with an assumed albedo and no atmosphere.
    teq_centre_k: float = 255.0
    teq_sigma_k: float = 55.0

    #: Host-star temperature preference. Deliberately broad: M dwarfs are
    #: legitimate hosts, so the penalty is mild rather than exclusionary.
    host_teff_centre_k: float = 5500.0
    host_teff_sigma_k: float = 1600.0

    use_optimistic_hz: bool = False

    def weights(self) -> dict[str, float]:
        return {
            "radius": self.w_radius,
            "incident_flux": self.w_flux,
            "temperature": self.w_temperature,
            "habitable_zone": self.w_habitable_zone,
            "host_star": self.w_host_star,
        }

    def normalised_weights(self) -> dict[str, float]:
        w = self.weights()
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total else w


DEFAULT_CONFIG = ScoringConfig()


# ---------------------------------------------------------------------------
# Component scoring functions. Each returns 0-1, vectorised, NaN-safe.
# ---------------------------------------------------------------------------
def _gauss(x: np.ndarray, centre: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((np.asarray(x, dtype=float) - centre) / sigma) ** 2)


def score_radius(radius_re: np.ndarray, cfg: ScoringConfig = DEFAULT_CONFIG) -> np.ndarray:
    """
    Log-space Gaussian centred on 1 R_earth, with a soft rocky-ceiling roll-off.

    Log space because radius errors are multiplicative: a factor-of-2 error is
    equally likely upward and downward, which a linear Gaussian does not model.
    """
    r = np.asarray(radius_re, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = _gauss(np.log10(np.where(r > 0, r, np.nan)), 0.0, cfg.radius_log_sigma)
    # Soft roll-off above the rocky ceiling: a logistic, not a cliff.
    ceiling = 1.0 / (1.0 + np.exp((r - cfg.rocky_ceiling_re) / 0.25))
    return np.clip(base * ceiling, 0.0, 1.0)


def score_flux(insolation_earth: np.ndarray, cfg: ScoringConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Log-space Gaussian on incident flux, centred on Earth's 1.0."""
    s = np.asarray(insolation_earth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(_gauss(np.log10(np.where(s > 0, s, np.nan)), 0.0, cfg.flux_log_sigma),
                       0.0, 1.0)


def score_temperature(teq_k: np.ndarray, cfg: ScoringConfig = DEFAULT_CONFIG) -> np.ndarray:
    """
    Gaussian on equilibrium temperature.

    koi_teq assumes a Bond albedo of 0.3, blackbody star and planet, and even
    day/night heat redistribution. It is not a surface temperature: Earth's
    equilibrium temperature is ~255 K against a 288 K mean surface, and the
    ~33 K difference is the greenhouse effect, which this quantity does not
    model. The component is weighted lowest of the physical terms for that
    reason, and it partly duplicates the flux term by construction.
    """
    return np.clip(_gauss(teq_k, cfg.teq_centre_k, cfg.teq_sigma_k), 0.0, 1.0)


def score_habitable_zone(
    insolation_earth: np.ndarray, teff_k: np.ndarray, cfg: ScoringConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """
    Smooth membership of the Kopparapu habitable zone.

    1.0 anywhere inside the zone, falling off smoothly outside it in log-flux
    space. Log-flux because flux goes as 1/d^2, so equal flux steps are very
    unequal distance steps.
    """
    s = np.asarray(insolation_earth, dtype=float)
    outer, inner = hz_flux_bounds(teff_k, cfg.use_optimistic_hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_s, log_in, log_out = np.log10(s), np.log10(inner), np.log10(outer)
        width = np.abs(log_in - log_out)
        # Distance outside the zone, in units of the zone's own log-width.
        excess = np.maximum(np.maximum(log_s - log_in, log_out - log_s), 0.0)
        return np.clip(np.exp(-0.5 * (excess / (0.5 * width)) ** 2), 0.0, 1.0)


def score_host_star(
    teff_k: np.ndarray, srad_sun: np.ndarray, smass_sun: np.ndarray | None = None,
    cfg: ScoringConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Host-star suitability. Conservative and deliberately broad.

    A Sun-like star is not the only suitable host — the majority of stars are M
    dwarfs, and they host abundant small planets — so this is a mild preference,
    not a filter. The main-sequence term penalises evolved stars, whose inflated
    radii both distort the derived planet radius and imply a habitable zone that
    has migrated over the star's lifetime.
    """
    teff_term = _gauss(teff_k, cfg.host_teff_centre_k, cfg.host_teff_sigma_k)
    # Rough main-sequence radius expectation; penalise large departures.
    r = np.asarray(srad_sun, dtype=float)
    t = np.asarray(teff_k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_r = np.clip((t / 5772.0) ** 1.1, 0.1, 3.0)
        ratio = np.log10(np.where(r > 0, r, np.nan) / expected_r)
    ms_term = _gauss(ratio, 0.0, 0.28)
    # 0.65 floor: an unusual host reduces the score, it does not zero it.
    return np.clip(0.65 * teff_term + 0.35 * ms_term, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
@dataclass
class EarthLikeResult:
    earth_like_score: float
    band: str
    components: dict[str, float] = field(default_factory=dict)
    confidence_interval: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    n_monte_carlo: int = 0
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _components(
    radius_re, insol, teq, teff, srad, smass, cfg: ScoringConfig
) -> dict[str, np.ndarray]:
    return {
        "radius": score_radius(radius_re, cfg),
        "incident_flux": score_flux(insol, cfg),
        "temperature": score_temperature(teq, cfg),
        "habitable_zone": score_habitable_zone(insol, teff, cfg),
        "host_star": score_host_star(teff, srad, smass, cfg),
    }


def _combine(components: dict[str, np.ndarray], cfg: ScoringConfig) -> np.ndarray:
    """
    Weighted mean over AVAILABLE components, renormalised.

    A missing component must not silently score zero — that would rank a
    candidate with an unmeasured equilibrium temperature below an identical one
    where it happens to be catalogued. Missingness is instead surfaced as a
    warning and reflected in the data-quality score used for ranking.
    """
    weights = cfg.normalised_weights()
    stacked = np.vstack([np.atleast_1d(components[k]) for k in weights])
    w = np.array([weights[k] for k in weights])[:, None]
    available = np.isfinite(stacked)
    weighted = np.where(available, stacked * w, 0.0).sum(axis=0)
    total_w = np.where(available, w, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total_w > 0, weighted / total_w, np.nan) * 100.0


def compute_earth_like_score(
    *,
    planet_radius_re: float,
    insolation_earth: float | None = None,
    equilibrium_temp_k: float | None = None,
    stellar_teff_k: float | None = None,
    stellar_radius_sun: float | None = None,
    stellar_mass_sun: float | None = None,
    orbital_period_days: float | None = None,
    planet_radius_err: float | None = None,
    stellar_radius_err: float | None = None,
    stellar_teff_err: float | None = None,
    insolation_err: float | None = None,
    config: ScoringConfig = DEFAULT_CONFIG,
    n_samples: int = 0,
    seed: int = 42,
) -> EarthLikeResult:
    """
    Score one candidate, optionally with Monte Carlo uncertainty propagation.

    `n_samples=0` gives the point estimate only (fast path, used by the API's
    FAST_MODE). Any positive value propagates the supplied 1-sigma errors and
    returns a 5th-95th percentile interval.
    """
    warnings_: list[str] = []
    cfg = config

    if insolation_earth is None and None not in (stellar_radius_sun, stellar_teff_k,
                                                 orbital_period_days, stellar_mass_sun):
        a_au = float(C.semimajor_axis_au(orbital_period_days, stellar_mass_sun))
        insolation_earth = float(C.insolation_earth_units(stellar_radius_sun,
                                                          stellar_teff_k, a_au))
        warnings_.append("Incident flux was derived from stellar parameters and "
                         "orbital period, not taken from the catalogue.")

    if equilibrium_temp_k is None and insolation_earth is not None:
        # T_eq = T_earth * S^(1/4), from the same albedo-0.3 blackbody assumption.
        equilibrium_temp_k = float(C.T_EQ_EARTH_K * max(insolation_earth, 1e-9) ** 0.25)
        warnings_.append("Equilibrium temperature was derived from incident flux.")

    point = _components(
        np.array([planet_radius_re], dtype=float),
        np.array([insolation_earth if insolation_earth is not None else np.nan]),
        np.array([equilibrium_temp_k if equilibrium_temp_k is not None else np.nan]),
        np.array([stellar_teff_k if stellar_teff_k is not None else np.nan]),
        np.array([stellar_radius_sun if stellar_radius_sun is not None else np.nan]),
        np.array([stellar_mass_sun if stellar_mass_sun is not None else np.nan]),
        cfg,
    )
    score = float(_combine(point, cfg)[0])

    components = {k: (float(np.round(v[0] * 100, 1)) if np.isfinite(v[0]) else None)
                  for k, v in point.items()}
    for name, value in components.items():
        if value is None:
            warnings_.append(f"Component '{name}' could not be computed from the "
                             "supplied parameters and was excluded from the weighted mean.")

    # --- warnings that qualify the number ----------------------------------
    if planet_radius_err and planet_radius_re:
        frac = abs(planet_radius_err) / planet_radius_re
        if frac > 0.20:
            warnings_.append(
                f"Planet radius uncertainty is high ({frac:.0%}). Planet radius is "
                "derived from the stellar radius, so this reflects how well the host "
                "star is characterised."
            )
    if stellar_teff_k is not None and not (TEFF_VALID_MIN_K <= stellar_teff_k <= TEFF_VALID_MAX_K):
        warnings_.append(
            f"Host effective temperature {stellar_teff_k:.0f} K is outside the "
            f"{TEFF_VALID_MIN_K:.0f}-{TEFF_VALID_MAX_K:.0f} K validity range of the "
            "habitable-zone parametrisation; that component is an extrapolation."
        )
    warnings_.append("Equilibrium temperature is not surface temperature: it assumes "
                     "a Bond albedo of 0.3 and models no atmosphere.")
    warnings_.append("The habitable zone is a liquid-water proxy under stated "
                     "atmospheric assumptions, not evidence of habitability.")

    result = EarthLikeResult(
        earth_like_score=round(score, 1) if np.isfinite(score) else float("nan"),
        band=band_for(score) if np.isfinite(score) else "Unscored",
        components=components,
        warnings=warnings_,
    )

    if n_samples > 0:
        lo, med, hi, mean = _monte_carlo(
            planet_radius_re, insolation_earth, equilibrium_temp_k,
            stellar_teff_k, stellar_radius_sun, stellar_mass_sun,
            planet_radius_err, insolation_err, stellar_teff_err, stellar_radius_err,
            cfg, n_samples, seed,
        )
        result.confidence_interval = {
            "lower": round(lo, 1), "median": round(med, 1),
            "upper": round(hi, 1), "mean": round(mean, 1),
            "interval": "5th-95th percentile",
        }
        result.n_monte_carlo = n_samples
        if np.isfinite(hi - lo) and (hi - lo) > 30:
            result.warnings.insert(
                0, f"Score is poorly constrained: the 5th-95th percentile range spans "
                   f"{hi - lo:.0f} points. Treat the point estimate as indicative only.")
    return result


def _monte_carlo(
    prad, insol, teq, teff, srad, smass,
    prad_err, insol_err, teff_err, srad_err,
    cfg: ScoringConfig, n: int, seed: int,
) -> tuple[float, float, float, float]:
    """
    Propagate measurement uncertainty by resampling the inputs.

    Radius and flux are sampled in LOG space (lognormal), because their errors
    are multiplicative and a normal draw would put non-trivial mass at negative
    radius for the ~30% fractional errors typical of KIC-provenance rows.
    """
    rng = np.random.default_rng(seed)

    def draw_log(value, err, default_frac=0.10):
        if value is None or not np.isfinite(value) or value <= 0:
            return np.full(n, np.nan)
        frac = abs(err) / value if err else default_frac
        return value * np.exp(rng.normal(0.0, np.log1p(min(frac, 2.0)), n))

    def draw_lin(value, err, default_frac=0.05):
        if value is None or not np.isfinite(value):
            return np.full(n, np.nan)
        sigma = abs(err) if err else abs(value) * default_frac
        return rng.normal(value, sigma, n)

    prad_s = draw_log(prad, prad_err, 0.30)        # 30% default: the KIC median
    srad_s = draw_log(srad, srad_err, 0.25)
    teff_s = draw_lin(teff, teff_err, 0.035)       # ~200 K on a 5700 K star
    insol_s = draw_log(insol, insol_err, 0.35)     # inherits stellar radius + Teff^4
    smass_s = np.full(n, smass if smass is not None else np.nan)

    if teq is not None and np.isfinite(teq):
        # Recompute T_eq from the sampled flux so the two stay physically
        # coupled: T_eq scales as S^(1/4), so sampling them independently would
        # produce combinations that violate the relation they were derived from.
        base = insol if (insol is not None and np.isfinite(insol) and insol > 0) else None
        if base is not None:
            with np.errstate(invalid="ignore"):
                teq_s = teq * (insol_s / base) ** 0.25
        else:
            teq_s = np.full(n, float(teq))
    else:
        teq_s = np.full(n, np.nan)

    comps = _components(prad_s, insol_s, teq_s, teff_s, srad_s, smass_s, cfg)
    scores = _combine(comps, cfg)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return (float("nan"),) * 4
    return (float(np.percentile(finite, 5)), float(np.median(finite)),
            float(np.percentile(finite, 95)), float(finite.mean()))
