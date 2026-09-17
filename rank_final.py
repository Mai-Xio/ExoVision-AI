import pandas as pd
import numpy as np
from pathlib import Path
import joblib


# ============================================================
# PATHS
# ============================================================

DATASET = Path("koi_training_data.csv")
MODEL = Path("artifacts_final/random_forest_final.joblib")
OUTPUT_DIR = Path("artifacts_final")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA + MODEL
# ============================================================

df = pd.read_csv(DATASET)

pipeline = joblib.load(MODEL)


# ============================================================
# FINAL MODEL FEATURES
# ============================================================

feature_list_file = OUTPUT_DIR / "feature_list.json"

import json

with open(feature_list_file, "r", encoding="utf-8") as f:
    feature_list = json.load(f)

if isinstance(feature_list, list):
    features = feature_list
else:
    features = feature_list["features"]

# ============================================================
# PREDICTIONS
# ============================================================

X = df[features]

df["planet_probability"] = pipeline.predict_proba(X)[:, 1]


# ============================================================
# EARTH-LIKE SCREEN
# ============================================================

screen = (
    df["koi_prad"].between(0.8, 1.25)
    & df["koi_insol"].between(0.5, 2.0)
    & (df["planet_probability"] >= 0.80)
)

candidates = df.loc[screen].copy()


# ============================================================
# PROJECT-DEFINED EARTH-LIKE SCREENING SCORE
# ============================================================

def gaussian_score(value, target, scale):
    return np.exp(
        -0.5 * ((value - target) / scale) ** 2
    )


def log_gaussian_score(value, target, scale):
    return np.exp(
        -0.5 * (
            np.log10(value / target) / scale
        ) ** 2
    )


# Radius similarity to Earth
candidates["radius_score"] = gaussian_score(
    candidates["koi_prad"],
    1.0,
    0.30,
)


# Insolation similarity to Earth
candidates["insolation_score"] = log_gaussian_score(
    candidates["koi_insol"],
    1.0,
    np.log10(2),
)


# Approximate equilibrium-temperature similarity
candidates["temperature_score"] = gaussian_score(
    candidates["koi_teq"],
    255,
    80,
)


# Stellar-temperature similarity to Sun
candidates["stellar_temperature_score"] = gaussian_score(
    candidates["koi_steff"],
    5772,
    1000,
)


# Orbital-period similarity to Earth
candidates["period_score"] = log_gaussian_score(
    candidates["koi_period"],
    365.25,
    0.8,
)


# ============================================================
# PHYSICS CONSISTENCY
#
# IMPORTANT:
# These definitions MUST match validate_candidates.py
# ============================================================

candidates["density_consistent"] = (
    candidates["f_rho_ratio"].between(0.7, 1.3)
)

candidates["duration_consistent"] = (
    candidates["f_duration_ratio"].between(0.5, 1.5)
    & ~candidates["f_duration_unphysical"]
)

candidates["depth_consistent"] = (
    candidates["f_depth_ratio"].between(0.7, 1.3)
)

candidates["overall_physics_consistent"] = (
    candidates["density_consistent"]
    & candidates["duration_consistent"]
    & candidates["depth_consistent"]
)


# ============================================================
# TRANSIT CONSISTENCY SCORE
# ============================================================

transit_consistency_score = (
    np.exp(-candidates["f_log_rho_ratio"].abs())
    * np.exp(-candidates["f_impact_discrepancy"].abs())
    * np.exp(-np.log(candidates["f_depth_ratio"]).abs())
)

candidates["transit_consistency_score"] = (
    transit_consistency_score.clip(0, 1)
)


# ============================================================
# FINAL PROJECT SCREENING SCORE
# ============================================================

candidates["earthlike_score"] = (
    0.30 * candidates["planet_probability"]
    + 0.20 * candidates["radius_score"]
    + 0.20 * candidates["insolation_score"]
    + 0.10 * candidates["temperature_score"]
    + 0.05 * candidates["stellar_temperature_score"]
    + 0.05 * candidates["period_score"]
    + 0.10 * candidates["transit_consistency_score"]
)


# ============================================================
# RANK
# ============================================================

candidates = candidates.sort_values(
    "earthlike_score",
    ascending=False,
).reset_index(drop=True)

candidates["final_rank"] = (
    candidates.index + 1
)


# ============================================================
# CATALOG STATUS
# ============================================================

candidates["catalog_status"] = (
    candidates["koi_disposition"]
    .map(
        {
            "CONFIRMED": "CONFIRMED",
            "CANDIDATE": "CANDIDATE",
            "FALSE POSITIVE": "FALSE_POSITIVE",
        }
    )
    .fillna("OTHER")
)


# ============================================================
# OUTPUT TABLE
# ============================================================

output_columns = [
    "final_rank",
    "kepoi_name",
    "kepid",
    "planet_probability",
    "earthlike_score",
    "catalog_status",
    "koi_prad",
    "koi_insol",
    "koi_teq",
    "koi_period",
    "koi_steff",
    "koi_srad",
    "koi_smass",
    "f_rho_ratio",
    "f_duration_ratio",
    "f_duration_unphysical",
    "f_depth_ratio",
    "f_log_rho_ratio",
    "f_impact_discrepancy",
    "density_consistent",
    "duration_consistent",
    "depth_consistent",
    "overall_physics_consistent",
]

output_columns = [
    col for col in output_columns
    if col in candidates.columns
]

final_candidates = candidates[output_columns].copy()


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR / "final_earthlike_candidates.csv"
)

final_candidates.to_csv(
    output_file,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 78)
print("EXOVISION AI — FINAL EARTH-LIKE SCREENING")
print("=" * 78)

print()
print(f"Total dataset rows: {len(df)}")
print("Radius range: 0.8–1.25 Earth radii")
print("Insolation range: 0.5–2.0 Earth flux")
print("Minimum planet probability: 0.80")
print(f"Candidates passing screen: {len(final_candidates)}")

print()
print("PHYSICS CONSISTENCY")
print("-" * 78)

print(
    "Density consistent:",
    int(final_candidates["density_consistent"].sum()),
    "/",
    len(final_candidates),
)

print(
    "Duration consistent:",
    int(final_candidates["duration_consistent"].sum()),
    "/",
    len(final_candidates),
)

print(
    "Depth consistent:",
    int(final_candidates["depth_consistent"].sum()),
    "/",
    len(final_candidates),
)

print(
    "Overall physics consistent:",
    int(final_candidates["overall_physics_consistent"].sum()),
    "/",
    len(final_candidates),
)

print()

if len(final_candidates) > 0:
    print(
        "Highest screening score:",
        f"{final_candidates['earthlike_score'].iloc[0]:.4f}",
    )

print()
print("RANKED CANDIDATES")
print("-" * 78)

display_cols = [
    "final_rank",
    "kepoi_name",
    "planet_probability",
    "earthlike_score",
    "koi_prad",
    "koi_insol",
    "koi_teq",
    "density_consistent",
    "duration_consistent",
    "depth_consistent",
    "overall_physics_consistent",
]

print(
    final_candidates[display_cols]
    .to_string(index=False)
)

print()
print("Saved:")
print(output_file)

print("=" * 78)
print("FINAL CANDIDATE RANKING COMPLETE")
print("=" * 78)