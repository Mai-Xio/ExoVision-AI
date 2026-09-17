import json
from pathlib import Path

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
OUTPUT_DIR = Path("artifacts_ablation")
OUTPUT_DIR.mkdir(exist_ok=True)


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


FAMILIES = {
    "density": [
        "f_rho_cat_from_logg",
        "f_rho_star_catalogue",
        "f_rho_ratio",
        "f_log_rho_ratio",
        "f_abs_log_rho_ratio",
    ],
    "depth": [
        "f_depth_expected_ppm",
        "f_depth_ratio",
        "f_log_depth",
    ],
    "radius": [
        "f_log_prad",
        "f_ror_from_radii",
        "f_ror_discrepancy",
    ],
    "duration": [
        "f_duration_ratio",
        "f_duration_unphysical",
        "f_impact_implied",
        "f_impact_discrepancy",
    ],
}


def metrics(y_true, y_pred, y_prob):
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


def evaluate_model(model, X, y, groups, split_mask):
    X_train = X[split_mask]
    y_train = y[split_mask]
    groups_train = groups[split_mask]

    X_test = X[~split_mask]
    y_test = y[~split_mask]

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_rows = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train, groups_train), start=1
    ):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]

        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)

        val_prob = model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)

        fold_metrics = metrics(y_val, val_pred, val_prob)
        fold_metrics["fold"] = fold

        cv_rows.append(fold_metrics)

    cv_df = pd.DataFrame(cv_rows)

    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    test_metrics = metrics(y_test, test_pred, test_prob)

    return cv_df, test_metrics, model


def main():
    df = pd.read_csv(DATA_PATH)

    y = df["target_binary"]
    groups = df["kepid"]

    split_mask = df["split"].eq("train")

    constant_features = [
        c for c in ALL_FEATURES
        if df[c].nunique(dropna=True) <= 1
    ]

    experiments = {
        "full": ALL_FEATURES.copy(),
        "no_density": [
            c for c in ALL_FEATURES
            if c not in FAMILIES["density"]
        ],
        "no_depth": [
            c for c in ALL_FEATURES
            if c not in FAMILIES["depth"]
        ],
        "no_radius": [
            c for c in ALL_FEATURES
            if c not in FAMILIES["radius"]
        ],
        "no_duration": [
            c for c in ALL_FEATURES
            if c not in FAMILIES["duration"]
        ],
        "no_constants": [
            c for c in ALL_FEATURES
            if c not in constant_features
        ],
    }

    print("=" * 80)
    print("FEATURE-FAMILY ABLATION EXPERIMENT")
    print("=" * 80)
    print()

    print(f"Rows: {len(df)}")
    print(f"Training rows: {split_mask.sum()}")
    print(f"Test rows: {(~split_mask).sum()}")
    print(f"Total features: {len(ALL_FEATURES)}")
    print()

    print("Constant features removed by no_constants:")
    for feature in constant_features:
        print(f"  - {feature}")
    print()

    all_results = []

    for experiment_name, features in experiments.items():
        print()
        print("-" * 80)
        print(f"EXPERIMENT: {experiment_name}")
        print(f"Features: {len(features)}")
        print("-" * 80)

        for model_name, builder in [
            ("random_forest", build_rf),
            ("gradient_boosting", build_gb),
        ]:
            print(f"\nModel: {model_name}")

            model = builder()

            X = df[features]

            cv_df, test_metrics, fitted_model = evaluate_model(
                model,
                X,
                y,
                groups,
                split_mask,
            )

            cv_mean = cv_df.drop(columns=["fold"]).mean()

            row = {
                "experiment": experiment_name,
                "model": model_name,
                "n_features": len(features),
            }

            for metric_name, value in cv_mean.items():
                row[f"cv_{metric_name}"] = value

            for metric_name, value in test_metrics.items():
                row[f"test_{metric_name}"] = value

            all_results.append(row)

            print(
                f"CV ROC-AUC: {cv_mean['roc_auc']:.4f} | "
                f"PR-AUC: {cv_mean['pr_auc']:.4f} | "
                f"Brier: {cv_mean['brier']:.4f}"
            )

            print(
                f"Test ROC-AUC: {test_metrics['roc_auc']:.4f} | "
                f"PR-AUC: {test_metrics['pr_auc']:.4f} | "
                f"Brier: {test_metrics['brier']:.4f}"
            )

    results = pd.DataFrame(all_results)

    results.to_csv(
        OUTPUT_DIR / "ablation_results.csv",
        index=False,
    )

    with open(OUTPUT_DIR / "ablation_metadata.json", "w") as f:
        json.dump(
            {
                "features": ALL_FEATURES,
                "families": FAMILIES,
                "constant_features": constant_features,
                "experiments": {
                    name: features
                    for name, features in experiments.items()
                },
            },
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("FINAL ABLATION SUMMARY")
    print("=" * 80)

    display_columns = [
        "experiment",
        "model",
        "n_features",
        "cv_roc_auc",
        "cv_pr_auc",
        "cv_brier",
        "test_roc_auc",
        "test_pr_auc",
        "test_brier",
        "test_mcc",
    ]

    print(
        results[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_DIR / 'ablation_results.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'ablation_metadata.json'}")


if __name__ == "__main__":
    main()