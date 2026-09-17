from pathlib import Path

import numpy as np
import pandas as pd
import joblib


DATA_PATH = Path("koi_training_data.csv")
MODEL_PATH = Path(
    "artifacts_physics_grouped/random_forest.joblib"
)
OUTPUT_DIR = Path("artifacts_physics_grouped")


# ------------------------------------------------------------
# Earth reference values
# ------------------------------------------------------------

EARTH_RADIUS = 1.0
EARTH_INSOLATION = 1.0
EARTH_TEQ = 255.0
EARTH_STEFF = 5772.0
EARTH_PERIOD = 365.25


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("=" * 78)
print("EXOVISION AI — EARTH-LIKE CANDIDATE RANKING")
print("=" * 78)

df = pd.read_csv(DATA_PATH)

print(f"Dataset rows: {len(df)}")


# ------------------------------------------------------------
# Load trained physics-informed classifier
# ------------------------------------------------------------

model = joblib.load(MODEL_PATH)

physics_features = [
    c for c in df.columns
    if c.startswith("f_")
]

print(f"Physics features: {len(physics_features)}")


# ------------------------------------------------------------
# Calculate ML probability
# ------------------------------------------------------------

X = df[physics_features]

df["planet_probability"] = model.predict_proba(X)[:, 1]


# ------------------------------------------------------------
# Physical validity checks
# ------------------------------------------------------------

df["radius_score"] = np.exp(
    -0.5 *
    (
        (df["koi_prad"] - EARTH_RADIUS)
        / 0.30
    ) ** 2
)

df["insolation_score"] = np.exp(
    -0.5 *
    (
        np.log10(
            df["koi_insol"].clip(lower=1e-6)
        )
        /
        np.log10(2.0)
    ) ** 2
)

df["temperature_score"] = np.exp(
    -0.5 *
    (
        (df["koi_teq"] - EARTH_TEQ)
        / 80.0
    ) ** 2
)

df["stellar_temperature_score"] = np.exp(
    -0.5 *
    (
        (df["koi_steff"] - EARTH_STEFF)
        / 1000.0
    ) ** 2
)

df["period_score"] = np.exp(
    -0.5 *
    (
        np.log10(
            df["koi_period"].clip(lower=0.01)
            /
            EARTH_PERIOD
        )
        / 0.8
    ) ** 2
)


# ------------------------------------------------------------
# Transit / physics consistency
# ------------------------------------------------------------

df["transit_consistency_score"] = (
    np.exp(
        -np.abs(df["f_log_rho_ratio"])
    )
    *
    np.exp(
        -np.abs(df["f_impact_discrepancy"])
    )
    *
    np.exp(
        -np.abs(
            np.log(
                df["f_depth_ratio"].clip(lower=1e-6)
            )
        )
    )
)

df["transit_consistency_score"] = (
    df["transit_consistency_score"]
    .clip(0, 1)
)


# ------------------------------------------------------------
# Combine physical scores
# ------------------------------------------------------------

df["earthlike_score"] = (
    0.30 * df["planet_probability"]
    +
    0.20 * df["radius_score"]
    +
    0.20 * df["insolation_score"]
    +
    0.10 * df["temperature_score"]
    +
    0.05 * df["stellar_temperature_score"]
    +
    0.05 * df["period_score"]
    +
    0.10 * df["transit_consistency_score"]
)


# ------------------------------------------------------------
# Preliminary Earth-like physical screen
# ------------------------------------------------------------

screen = (
    df["koi_prad"].between(0.8, 1.25)
    &
    df["koi_insol"].between(0.5, 2.0)
    &
    df["planet_probability"].ge(0.80)
)


candidates = df.loc[screen].copy()


# ------------------------------------------------------------
# Rank candidates
# ------------------------------------------------------------

candidates = candidates.sort_values(
    "earthlike_score",
    ascending=False
).reset_index(drop=True)

candidates["final_rank"] = (
    np.arange(len(candidates)) + 1
)


# ------------------------------------------------------------
# Select output columns
# ------------------------------------------------------------

output_columns = [
    "final_rank",
    "kepoi_name",
    "kepid",
    "planet_probability",
    "earthlike_score",

    "radius_score",
    "insolation_score",
    "temperature_score",
    "stellar_temperature_score",
    "period_score",
    "transit_consistency_score",

    "koi_period",
    "koi_prad",
    "koi_insol",
    "koi_teq",
    "koi_steff",
    "koi_srad",
    "koi_smass",

    "f_sma_au",
    "f_luminosity_sun",
    "f_duration_ratio",
    "f_impact_discrepancy",
    "f_depth_ratio",
    "f_log_rho_ratio",
]


output_columns = [
    c for c in output_columns
    if c in candidates.columns
]


result = candidates[output_columns]


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)

output_path = (
    OUTPUT_DIR /
    "earthlike_candidates.csv"
)

result.to_csv(
    output_path,
    index=False
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("EARTH-LIKE CANDIDATE SCREEN")
print("=" * 78)

print(
    f"Initial physical candidates: "
    f"{((df['koi_prad'].between(0.8, 1.25)) & (df['koi_insol'].between(0.5, 2.0))).sum()}"
)

print(
    f"After ML probability >= 0.80: "
    f"{len(result)}"
)

if len(result) > 0:

    print("\nRanked candidates:")

    display_columns = [
        "final_rank",
        "kepoi_name",
        "planet_probability",
        "earthlike_score",
        "koi_period",
        "koi_prad",
        "koi_insol",
        "koi_teq",
        "koi_steff",
    ]

    print(
        result[display_columns]
        .to_string(index=False)
    )

else:

    print("\nNo candidates passed the combined screen.")


print("\nSaved:")
print(output_path.resolve())

print("=" * 78)