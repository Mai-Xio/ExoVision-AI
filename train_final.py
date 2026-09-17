from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
)


DATA_PATH = Path("koi_training_data.csv")
OUTPUT_DIR = Path("artifacts_final")
OUTPUT_DIR.mkdir(exist_ok=True)


PHYSICS_FEATURES = [
    "f_rho_cat_from_logg",
    "f_rho_star_catalogue",
    "f_rho_ratio",
    "f_log_rho_ratio",
    "f_abs_log_rho_ratio",
    "f_duration_max_hours",
    "f_duration_ratio",
    "f_duration_unphysical",
    "f_impact_implied",
    "f_impact_discrepancy",
    "f_depth_expected_ppm",
    "f_depth_ratio",
    "f_sma_au",
    "f_luminosity_sun",
    "f_insol_calc",
    "f_insol_ratio",
    "f_insol_filled",
    "f_teq_calc",
    "f_teq_ratio",
    "f_teq_filled",
    "f_prad_frac_err",
    "f_period_frac_err",
    "f_depth_frac_err",
    "f_srad_frac_err",
    "f_steff_frac_err",
    "f_duration_frac_err",
    "f_log_period",
    "f_log_depth",
    "f_log_prad",
    "f_log_insol",
    "f_log_model_snr",
    "f_log_max_mult_ev",
    "f_log_duration",
    "f_log_srho",
    "f_duty_cycle",
    "f_snr_per_transit",
    "f_ror_from_radii",
    "f_ror_discrepancy",
]


TARGET = "target_binary"
GROUP = "kepid"
SPLIT = "split"


def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
    }


def build_rf():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                    max_features="sqrt",
                ),
            ),
        ]
    )


def build_gb():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )


print("=" * 70)
print("EXOVISION AI — FINAL MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

missing_features = [
    feature for feature in PHYSICS_FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing required physics features:\n"
        + "\n".join(missing_features)
    )

required_columns = [TARGET, GROUP, SPLIT]

missing_required = [
    column for column in required_columns
    if column not in df.columns
]

if missing_required:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_required)
    )


print("\nChecking constant features...")

constant_features = []

for feature in PHYSICS_FEATURES:
    if df[feature].nunique(dropna=True) <= 1:
        constant_features.append(feature)

if constant_features:
    print("Removing zero-variance features:")
    for feature in constant_features:
        print(f"  - {feature}")
else:
    print("No zero-variance features found.")


FINAL_FEATURES = [
    feature
    for feature in PHYSICS_FEATURES
    if feature not in constant_features
]

print(f"\nPhysics features before cleanup: {len(PHYSICS_FEATURES)}")
print(f"Final features used: {len(FINAL_FEATURES)}")


train_df = df[df[SPLIT] == "train"].copy()
test_df = df[df[SPLIT] == "test"].copy()

print("\nExisting dataset split:")
print(f"Training rows: {len(train_df)}")
print(f"Test rows:     {len(test_df)}")


train_hosts = set(train_df[GROUP].dropna())
test_hosts = set(test_df[GROUP].dropna())
shared_hosts = train_hosts.intersection(test_hosts)

print(f"Training host stars: {len(train_hosts)}")
print(f"Test host stars:     {len(test_hosts)}")
print(f"Shared host stars:   {len(shared_hosts)}")

if shared_hosts:
    raise ValueError(
        f"Data leakage detected: {len(shared_hosts)} host stars "
        "are shared between train and test."
    )

print("Host-star isolation check: PASSED")


X_train = train_df[FINAL_FEATURES]
y_train = train_df[TARGET].astype(int)
groups_train = train_df[GROUP]

X_test = test_df[FINAL_FEATURES]
y_test = test_df[TARGET].astype(int)


print("\n" + "=" * 70)
print("GROUPED CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


def evaluate_cv(model_name, builder):
    fold_results = []

    print(f"\nModel: {model_name}")

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train, groups_train),
        start=1,
    ):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]

        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        group_fold_train = groups_train.iloc[train_idx]
        group_val = groups_train.iloc[val_idx]

        shared = set(group_fold_train).intersection(set(group_val))

        if shared:
            raise ValueError(
                f"Fold {fold} leakage: {len(shared)} shared host stars."
            )

        model = builder()

        model.fit(X_fold_train, y_fold_train)

        y_prob = model.predict_proba(X_val)[:, 1]

        result = metrics(y_val, y_prob)
        result["fold"] = fold

        fold_results.append(result)

        print(
            f"Fold {fold}: "
            f"ROC-AUC={result['roc_auc']:.4f}, "
            f"PR-AUC={result['pr_auc']:.4f}, "
            f"F1={result['f1']:.4f}, "
            f"MCC={result['mcc']:.4f}"
        )

    results_df = pd.DataFrame(fold_results)

    mean_metrics = results_df.drop(columns=["fold"]).mean().to_dict()

    print("\nCV mean:")
    for key, value in mean_metrics.items():
        print(f"  {key}: {value:.4f}")

    return results_df, mean_metrics


rf_cv_folds, rf_cv_mean = evaluate_cv(
    "Random Forest",
    build_rf,
)

gb_cv_folds, gb_cv_mean = evaluate_cv(
    "Gradient Boosting",
    build_gb,
)


print("\n" + "=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

print("\nTraining final Random Forest on all training data...")

rf_final = build_rf()
rf_final.fit(X_train, y_train)

rf_test_prob = rf_final.predict_proba(X_test)[:, 1]
rf_test_metrics = metrics(y_test, rf_test_prob)

print("\nRandom Forest test metrics:")
for key, value in rf_test_metrics.items():
    print(f"  {key}: {value:.4f}")


print("\nTraining final Gradient Boosting on all training data...")

gb_final = build_gb()
gb_final.fit(X_train, y_train)

gb_test_prob = gb_final.predict_proba(X_test)[:, 1]
gb_test_metrics = metrics(y_test, gb_test_prob)

print("\nGradient Boosting test metrics:")
for key, value in gb_test_metrics.items():
    print(f"  {key}: {value:.4f}")


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame(
    [
        {
            "model": "random_forest",
            "cv_roc_auc": rf_cv_mean["roc_auc"],
            "cv_pr_auc": rf_cv_mean["pr_auc"],
            "cv_brier": rf_cv_mean["brier"],
            "cv_mcc": rf_cv_mean["mcc"],
            "test_accuracy": rf_test_metrics["accuracy"],
            "test_precision": rf_test_metrics["precision"],
            "test_recall": rf_test_metrics["recall"],
            "test_f1": rf_test_metrics["f1"],
            "test_roc_auc": rf_test_metrics["roc_auc"],
            "test_pr_auc": rf_test_metrics["pr_auc"],
            "test_balanced_accuracy": rf_test_metrics["balanced_accuracy"],
            "test_mcc": rf_test_metrics["mcc"],
            "test_brier": rf_test_metrics["brier"],
        },
        {
            "model": "gradient_boosting",
            "cv_roc_auc": gb_cv_mean["roc_auc"],
            "cv_pr_auc": gb_cv_mean["pr_auc"],
            "cv_brier": gb_cv_mean["brier"],
            "cv_mcc": gb_cv_mean["mcc"],
            "test_accuracy": gb_test_metrics["accuracy"],
            "test_precision": gb_test_metrics["precision"],
            "test_recall": gb_test_metrics["recall"],
            "test_f1": gb_test_metrics["f1"],
            "test_roc_auc": gb_test_metrics["roc_auc"],
            "test_pr_auc": gb_test_metrics["pr_auc"],
            "test_balanced_accuracy": gb_test_metrics["balanced_accuracy"],
            "test_mcc": gb_test_metrics["mcc"],
            "test_brier": gb_test_metrics["brier"],
        },
    ]
)

print(comparison.to_string(index=False))

comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)


print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

joblib.dump(
    rf_final,
    OUTPUT_DIR / "random_forest_final.joblib",
)

joblib.dump(
    gb_final,
    OUTPUT_DIR / "gradient_boosting_final.joblib",
)

print("Saved:")
print("  artifacts_final/random_forest_final.joblib")
print("  artifacts_final/gradient_boosting_final.joblib")
print("  artifacts_final/model_comparison.csv")


print("\nSaving test predictions...")

rf_predictions = test_df[
    ["kepid", "kepoi_name", TARGET]
].copy()

rf_predictions["planet_probability"] = rf_test_prob
rf_predictions["prediction"] = (
    rf_test_prob >= 0.5
).astype(int)

rf_predictions.to_csv(
    OUTPUT_DIR / "random_forest_test_predictions.csv",
    index=False,
)


gb_predictions = test_df[
    ["kepid", "kepoi_name", TARGET]
].copy()

gb_predictions["planet_probability"] = gb_test_prob
gb_predictions["prediction"] = (
    gb_test_prob >= 0.5
).astype(int)

gb_predictions.to_csv(
    OUTPUT_DIR / "gradient_boosting_test_predictions.csv",
    index=False,
)


metadata = {
    "dataset": str(DATA_PATH),
    "rows": int(len(df)),
    "training_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "physics_features_original": len(PHYSICS_FEATURES),
    "physics_features_final": len(FINAL_FEATURES),
    "removed_constant_features": constant_features,
    "group_column": GROUP,
    "target_column": TARGET,
    "split_column": SPLIT,
    "training_unique_hosts": len(train_hosts),
    "test_unique_hosts": len(test_hosts),
    "shared_hosts": len(shared_hosts),
    "cv": {
        "type": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
    },
    "models": {
        "random_forest": {
            "n_estimators": 500,
            "class_weight": "balanced",
            "max_features": "sqrt",
            "random_state": 42,
        },
        "gradient_boosting": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "random_state": 42,
        },
    },
    "rf_cv_mean": rf_cv_mean,
    "gb_cv_mean": gb_cv_mean,
    "rf_test": rf_test_metrics,
    "gb_test": gb_test_metrics,
}


with open(
    OUTPUT_DIR / "run_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)


with open(
    OUTPUT_DIR / "feature_list.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(FINAL_FEATURES, f, indent=2)


print("\nSaved:")
print("  artifacts_final/random_forest_test_predictions.csv")
print("  artifacts_final/gradient_boosting_test_predictions.csv")
print("  artifacts_final/run_metadata.json")
print("  artifacts_final/feature_list.json")


print("\n" + "=" * 70)
print("FINAL TRAINING COMPLETE")
print("=" * 70)

print("\nFinal feature count:", len(FINAL_FEATURES))
print("Constant features removed:", constant_features)

print("\nRandom Forest test ROC-AUC:")
print(f"{rf_test_metrics['roc_auc']:.4f}")

print("\nGradient Boosting test ROC-AUC:")
print(f"{gb_test_metrics['roc_auc']:.4f}")

print("\nAll final artifacts are in:")
print(OUTPUT_DIR.resolve())