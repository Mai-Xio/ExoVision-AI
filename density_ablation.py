from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


DATA_PATH = Path("koi_training_data.csv")

ALL_FEATURES = [
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

DENSITY_FEATURES = [
    "f_rho_cat_from_logg",
    "f_rho_star_catalogue",
    "f_rho_ratio",
    "f_log_rho_ratio",
    "f_abs_log_rho_ratio",
]

EXPERIMENTS = {
    "full": ALL_FEATURES,

    "no_density": [
        f for f in ALL_FEATURES
        if f not in DENSITY_FEATURES
    ],

    "no_abs_log_rho_ratio": [
        f for f in ALL_FEATURES
        if f != "f_abs_log_rho_ratio"
    ],

    "only_abs_log_rho_ratio": [
        f for f in ALL_FEATURES
        if f not in DENSITY_FEATURES
        or f == "f_abs_log_rho_ratio"
    ],

    "no_abs_but_keep_density": [
        f for f in ALL_FEATURES
        if f != "f_abs_log_rho_ratio"
    ],
}


def build_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def build_gb():
    return Pipeline([
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
    ])


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "brier": brier_score_loss(
            y_true, probabilities
        ),
        "mcc": matthews_corrcoef(
            y_true, predictions
        ),
    }


def run_experiment(df, features, model_name):
    train_mask = df["split"].eq("train")

    X = df[features]
    y = df["target_binary"]
    groups = df["kepid"]

    X_train = X[train_mask]
    y_train = y[train_mask]
    groups_train = groups[train_mask]

    X_test = X[~train_mask]
    y_test = y[~train_mask]

    if model_name == "random_forest":
        model = build_rf()
    else:
        model = build_gb()

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_results = []

    for train_idx, val_idx in cv.split(
        X_train,
        y_train,
        groups_train,
    ):
        model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
        )

        val_prob = model.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]

        fold_results.append(
            calculate_metrics(
                y_train.iloc[val_idx],
                val_prob,
            )
        )

    cv_df = pd.DataFrame(fold_results)

    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_prob,
    )

    return cv_df.mean(numeric_only=True), test_metrics


def main():

    df = pd.read_csv(DATA_PATH)

    print("=" * 80)
    print("TARGETED DENSITY ABLATION")
    print("=" * 80)

    print()
    print("Checking zero-variance features...")
    print()

    zero_variance = [
        c for c in ALL_FEATURES
        if df[c].nunique(dropna=True) <= 1
    ]

    if zero_variance:
        for feature in zero_variance:
            print(f"  {feature}")
    else:
        print("  None")

    print()
    print("=" * 80)
    print("EXPERIMENTS")
    print("=" * 80)

    results = []

    for experiment_name, features in EXPERIMENTS.items():

        print()
        print("-" * 80)
        print(
            f"{experiment_name}: "
            f"{len(features)} features"
        )
        print("-" * 80)

        for model_name in [
            "random_forest",
            "gradient_boosting",
        ]:

            print(
                f"\nRunning {model_name}..."
            )

            cv_metrics, test_metrics = run_experiment(
                df,
                features,
                model_name,
            )

            row = {
                "experiment": experiment_name,
                "model": model_name,
                "features": len(features),

                "cv_roc_auc": cv_metrics["roc_auc"],
                "cv_pr_auc": cv_metrics["pr_auc"],
                "cv_brier": cv_metrics["brier"],
                "cv_mcc": cv_metrics["mcc"],

                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_brier": test_metrics["brier"],
                "test_mcc": test_metrics["mcc"],
            }

            results.append(row)

            print(
                f"CV ROC-AUC:  {row['cv_roc_auc']:.4f}"
            )
            print(
                f"CV PR-AUC:   {row['cv_pr_auc']:.4f}"
            )
            print(
                f"CV Brier:    {row['cv_brier']:.4f}"
            )
            print(
                f"CV MCC:      {row['cv_mcc']:.4f}"
            )

            print(
                f"Test ROC-AUC:{row['test_roc_auc']:.4f}"
            )
            print(
                f"Test PR-AUC: {row['test_pr_auc']:.4f}"
            )
            print(
                f"Test Brier:  {row['test_brier']:.4f}"
            )
            print(
                f"Test MCC:    {row['test_mcc']:.4f}"
            )

    results_df = pd.DataFrame(results)

    output_dir = Path("artifacts_density_ablation")
    output_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        output_dir / "density_ablation_results.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        results_df.round(4).to_string(index=False)
    )

    print()
    print(
        "Saved:",
        output_dir / "density_ablation_results.csv",
    )


if __name__ == "__main__":
    main()