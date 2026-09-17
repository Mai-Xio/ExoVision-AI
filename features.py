"""
Central feature-set configuration.

Single source of truth for which columns each model may see. Feature lists are
never written inline in a training script — a list that exists in two places
will eventually disagree in one of them, and the disagreement that matters here
is the one that readmits a quarantined column.

The three sets
--------------
A  `baseline`         Directly measured / catalogued observational parameters.
B  `physics_informed` Baseline + the consistency features from Step 2 + QC flags.
                      THE PRIMARY SET.
C  `leaky_control`    Baseline + the quarantined Robovetter columns.
                      EDUCATIONAL ONLY. Never used for prediction, ranking,
                      deployment, or any headline result.

Resolution against the real schema
----------------------------------
KOI deliveries differ from each other: a column present in `cumulative` may be
absent from `q1_q17_dr25_koi`. Every list here is *requested*; `resolve()`
intersects it with the dataframe actually loaded and reports what was dropped.
Never assume a column exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QUARANTINE — the Robovetter's own verdict. See docs/01-science-brief.md.
#
#   koi_fpflag_* --(rule)--> koi_pdisposition --(copied)--> koi_disposition
#                                  ^
#                            koi_score
#
# Any model given these is recovering a deterministic function it was handed.
# ---------------------------------------------------------------------------
QUARANTINED: Final[tuple[str, ...]] = (
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_score",
    "koi_pdisposition",
    "koi_comment",
    "koi_disp_prov",
    "koi_vet_stat",
    "koi_vet_date",
    "kepler_name",        # non-null ~ CONFIRMED in practice
    "koi_datalink_dvr",
    "koi_datalink_dvs",
)

#: Columns that identify a row. Never features.
IDENTIFIERS: Final[tuple[str, ...]] = ("kepid", "kepoi_name", "kepler_name")

#: Target columns produced by cleaning. Never features.
TARGETS: Final[tuple[str, ...]] = (
    "koi_disposition", "target_binary", "target_multiclass",
)

#: Generator ground truth present only in synthetic data. Never a feature.
SYNTHETIC_ONLY: Final[tuple[str, ...]] = ("_truth_kind",)


# ---------------------------------------------------------------------------
# SET A — baseline observational features
# ---------------------------------------------------------------------------
BASELINE_TRANSIT: Final[tuple[str, ...]] = (
    "koi_period",       # days
    "koi_duration",     # hours
    "koi_depth",        # ppm
    "koi_ror",          # Rp/Rs, the directly fitted quantity
    "koi_prad",         # Earth radii (derived: koi_ror * koi_srad)
    "koi_impact",
    "koi_dor",          # a/Rs
    "koi_sma",          # au
    "koi_incl",
    "koi_srho",         # g/cm^3, fitted FROM the light curve
    "koi_teq",          # K, blackbody, albedo 0.3 assumed
    "koi_insol",        # Earth flux units
)

BASELINE_SIGNAL: Final[tuple[str, ...]] = (
    "koi_model_snr",
    "koi_max_mult_ev",
    "koi_max_sngle_ev",
    "koi_num_transits",
    "koi_count",
    "koi_model_chisq",
)

BASELINE_STELLAR: Final[tuple[str, ...]] = (
    "koi_steff", "koi_slogg", "koi_srad", "koi_smass", "koi_smet", "koi_kepmag",
)

#: Categorical provenance columns. One-hot encoded inside the pipeline.
CATEGORICAL: Final[tuple[str, ...]] = ("koi_sparprov", "koi_fittype")

BASELINE: Final[tuple[str, ...]] = (
    BASELINE_TRANSIT + BASELINE_SIGNAL + BASELINE_STELLAR + CATEGORICAL
)

# ---------------------------------------------------------------------------
# SET B — physics-informed additions (built by ml/preprocessing/features.py)
# ---------------------------------------------------------------------------
#: The three consistency diagnostics that replace the quarantined flags'
#: discriminating power using physics the archive publishes anyway.
CONSISTENCY_CORE: Final[tuple[str, ...]] = (
    "f_abs_log_rho_ratio",   # stellar density: light-curve fit vs catalogue
    "f_log_rho_ratio",
    "f_rho_ratio",
    "f_duration_ratio",      # observed vs geometric max; bounded by 1
    "f_depth_ratio",         # measured depth vs (Rp/Rs)^2
)

CONSISTENCY_SUPPORT: Final[tuple[str, ...]] = (
    "f_rho_star_catalogue", "f_duration_max_hours",
    "f_impact_implied", "f_impact_discrepancy",
    "f_depth_expected_ppm", "f_ror_from_radii", "f_ror_discrepancy",
    "f_duty_cycle", "f_snr_per_transit",
)

DERIVED_PHYSICAL: Final[tuple[str, ...]] = (
    "f_sma_au", "f_luminosity_sun", "f_insol_calc", "f_insol_ratio",
    "f_insol_filled", "f_teq_calc", "f_teq_ratio", "f_teq_filled",
)

UNCERTAINTY: Final[tuple[str, ...]] = (
    "f_prad_frac_err", "f_period_frac_err", "f_depth_frac_err",
    "f_srad_frac_err", "f_steff_frac_err", "f_duration_frac_err",
)

LOG_TRANSFORMS: Final[tuple[str, ...]] = (
    "f_log_period", "f_log_depth", "f_log_prad", "f_log_insol",
    "f_log_model_snr", "f_log_max_mult_ev", "f_log_duration", "f_log_srho",
)

#: Boolean quality-control flags produced by cleaning. Model-usable by design.
QC_FLAGS: Final[tuple[str, ...]] = (
    "qc_radius_implausible", "qc_depth_stellar", "qc_below_detection_snr",
    "qc_duration_too_long", "qc_grazing", "qc_stellar_params_assumed",
    "qc_no_transit_fit", "f_duration_unphysical", "f_rho_cat_from_logg",
)

PHYSICS_INFORMED: Final[tuple[str, ...]] = (
    BASELINE + CONSISTENCY_CORE + CONSISTENCY_SUPPORT
    + DERIVED_PHYSICAL + UNCERTAINTY + LOG_TRANSFORMS + QC_FLAGS
)

# ---------------------------------------------------------------------------
# SET C — leaky control. Educational only.
# ---------------------------------------------------------------------------
LEAKY_CONTROL: Final[tuple[str, ...]] = BASELINE + (
    "koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
)

FEATURE_SETS: Final[dict[str, tuple[str, ...]]] = {
    "baseline": BASELINE,
    "physics_informed": PHYSICS_INFORMED,
    "leaky_control": LEAKY_CONTROL,
}

#: Sets that may be used for any real scientific claim, model selection,
#: candidate ranking, or deployment.
VALID_FEATURE_SETS: Final[tuple[str, ...]] = ("baseline", "physics_informed")

#: The primary set.
PRIMARY_FEATURE_SET: Final[str] = "physics_informed"

FEATURE_SET_DESCRIPTIONS: Final[dict[str, str]] = {
    "baseline": "Directly measured and catalogued observational parameters only.",
    "physics_informed": (
        "Baseline plus physics-derived self-consistency diagnostics "
        "(stellar density, transit duration, transit depth), derived "
        "uncertainties, log transforms and quality-control flags."
    ),
    "leaky_control": (
        "LEAKY CONTROL — NOT VALID FOR SCIENTIFIC USE. Deliberately includes "
        "Robovetter-derived columns that are upstream of the label. Exists "
        "solely to quantify how much apparent performance target leakage buys."
    ),
}


@dataclass(frozen=True)
class ResolvedFeatures:
    """A feature set intersected with the schema actually present."""

    name: str
    numeric: list[str]
    categorical: list[str]
    boolean: list[str]
    missing: list[str]

    @property
    def all(self) -> list[str]:
        return self.numeric + self.categorical + self.boolean

    def __len__(self) -> int:
        return len(self.all)


def resolve(df: pd.DataFrame, feature_set: str) -> ResolvedFeatures:
    """
    Intersect a named feature set with the columns present in `df` and split
    them by dtype so the ColumnTransformer can route them correctly.

    Raises if a valid feature set has somehow acquired a quarantined column —
    a bug here is the one that silently invalidates every result downstream.
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {feature_set!r}; have {list(FEATURE_SETS)}")

    requested = list(dict.fromkeys(FEATURE_SETS[feature_set]))
    present = [c for c in requested if c in df.columns]
    missing = [c for c in requested if c not in df.columns]

    if feature_set in VALID_FEATURE_SETS:
        contamination = sorted(set(present) & set(QUARANTINED))
        if contamination:
            raise AssertionError(
                f"feature set {feature_set!r} contains quarantined columns: "
                f"{contamination}. This would invalidate every downstream result."
            )
    banned = sorted(set(present) & (set(TARGETS) | set(SYNTHETIC_ONLY)))
    if banned:
        raise AssertionError(f"feature set {feature_set!r} contains targets: {banned}")

    numeric, categorical, boolean = [], [], []
    for col in present:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            boolean.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            categorical.append(col)

    if missing:
        logger.info(
            "feature set %r: %d/%d columns present; absent from this delivery: %s",
            feature_set, len(present), len(requested), ", ".join(missing[:12]),
        )
    return ResolvedFeatures(feature_set, numeric, categorical, boolean, missing)


def assert_no_leakage(columns: list[str], feature_set: str) -> None:
    """Guard usable anywhere a column list crosses a module boundary."""
    if feature_set not in VALID_FEATURE_SETS:
        return
    bad = sorted(set(columns) & set(QUARANTINED))
    if bad:
        raise AssertionError(f"quarantined columns in {feature_set!r}: {bad}")
