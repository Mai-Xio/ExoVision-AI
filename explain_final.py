from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_PATH = Path("artifacts_final/random_forest_final.joblib")
DATA_PATH = Path("koi_training_data.csv")
OUTPUT_DIR = Path("artifacts_final")

OUTPUT_DIR.mkdir(exist_ok=True)


print("=" * 70)
print("EXOVISION AI — FINAL MODEL EXPLAINABILITY")
print("=" * 70)


print("\nLoading model...")
pipeline = joblib.load(MODEL_PATH)

print("Loading feature list...")
features = pd.read_json(
    OUTPUT_DIR / "feature_list.json",
    typ="series"
).tolist()

print(f"Features used by final model: {len(features)}")


print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

X = df[features]


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

estimator = pipeline.steps[-1][1]

importance = estimator.feature_importances_

importance_df = pd.DataFrame(
    {
        "feature": features,
        "importance": importance,
    }
).sort_values(
    "importance",
    ascending=False,
)


print("\n" + "=" * 70)
print("TOP 20 FEATURES")
print("=" * 70)

print(
    importance_df.head(20).to_string(index=False)
)


importance_df.to_csv(
    OUTPUT_DIR / "final_feature_importance.csv",
    index=False,
)


# ------------------------------------------------------------
# FEATURE IMPORTANCE PLOT
# ------------------------------------------------------------

top = importance_df.head(15).sort_values(
    "importance",
    ascending=True,
)

plt.figure(figsize=(10, 7))

plt.barh(
    top["feature"],
    top["importance"],
)

plt.xlabel("Random Forest feature importance")
plt.ylabel("Feature")
plt.title("ExoVision AI — Final Random Forest Feature Importance")

plt.tight_layout()

plot_path = OUTPUT_DIR / "final_feature_importance.png"

plt.savefig(
    plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved plot:")
print(plot_path)


# ------------------------------------------------------------
# IMPORTANCE BY PHYSICS FAMILY
# ------------------------------------------------------------

families = {
    "density": [
        "f_rho_cat_from_logg",
        "f_rho_star_catalogue",
        "f_rho_ratio",
        "f_log_rho_ratio",
        "f_abs_log_rho_ratio",
    ],

    "duration": [
        "f_duration_max_hours",
        "f_duration_ratio",
        "f_duration_unphysical",
        "f_impact_implied",
        "f_impact_discrepancy",
    ],

    "depth": [
        "f_depth_expected_ppm",
        "f_depth_ratio",
    ],

    "orbital": [
        "f_sma_au",
        "f_luminosity_sun",
        "f_insol_calc",
        "f_insol_filled",
        "f_teq_calc",
        "f_teq_filled",
    ],

    "uncertainty": [
        "f_prad_frac_err",
        "f_period_frac_err",
        "f_depth_frac_err",
        "f_srad_frac_err",
        "f_steff_frac_err",
        "f_duration_frac_err",
    ],

    "log_transforms": [
        "f_log_period",
        "f_log_depth",
        "f_log_prad",
        "f_log_insol",
        "f_log_model_snr",
        "f_log_max_mult_ev",
        "f_log_duration",
        "f_log_srho",
    ],

    "signal_quality": [
        "f_duty_cycle",
        "f_snr_per_transit",
    ],

    "radius_consistency": [
        "f_ror_from_radii",
        "f_ror_discrepancy",
    ],
}


family_rows = []

for family, family_features in families.items():

    family_features = [
        feature
        for feature in family_features
        if feature in importance_df["feature"].values
    ]

    family_importance = importance_df[
        importance_df["feature"].isin(family_features)
    ]["importance"].sum()

    family_rows.append(
        {
            "family": family,
            "total_importance": family_importance,
            "feature_count": len(family_features),
        }
    )


family_df = pd.DataFrame(
    family_rows
).sort_values(
    "total_importance",
    ascending=False,
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE BY PHYSICS FAMILY")
print("=" * 70)

print(
    family_df.to_string(index=False)
)


family_df.to_csv(
    OUTPUT_DIR / "final_feature_family_importance.csv",
    index=False,
)


# ------------------------------------------------------------
# CORRELATION AUDIT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET CORRELATION AUDIT")
print("=" * 70)

correlation_df = (
    df[features + ["target_binary"]]
    .corr(numeric_only=True)["target_binary"]
    .drop("target_binary")
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False,
    )
)

print(
    correlation_df.head(20).to_string()
)

correlation_df.to_csv(
    OUTPUT_DIR / "final_target_correlations.csv"
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPLAINABILITY AUDIT COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(
    "  artifacts_final/final_feature_importance.csv"
)

print(
    "  artifacts_final/final_feature_importance.png"
)

print(
    "  artifacts_final/final_feature_family_importance.csv"
)

print(
    "  artifacts_final/final_target_correlations.csv"
)