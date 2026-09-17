from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
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
OUT_DIR = Path("artifacts_physics_grouped")

TARGET = "target_binary"
GROUP = "kepid"


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


def print_metrics(title, result):
    print(f"\n{title}")

    for key, value in result.items():
        print(f"  {key:<20}: {value:.4f}")


def main():

    print("=" * 78)
    print("EXOVISION AI — PHYSICS FEATURES + GROUPED VALIDATION")
    print("=" * 78)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset: {DATA_PATH}")
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")

    required = [TARGET, GROUP, "split"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    # ------------------------------------------------------------
    # Explicitly select the physics-engineered feature layer.
    # ------------------------------------------------------------

    feature_cols = [
        c for c in df.columns
        if c.startswith("f_")
    ]

    if not feature_cols:
        raise ValueError("No f_* physics features were found.")

    print(f"\nPhysics-engineered features: {len(feature_cols)}")

    for col in feature_cols:
        print(f"  {col}")

    # ------------------------------------------------------------
    # Existing train/test split.
    # ------------------------------------------------------------

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("\nExisting split:")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows:     {len(test_df)}")

    # ------------------------------------------------------------
    # Verify host-star groups.
    # ------------------------------------------------------------

    train_groups = set(train_df[GROUP].dropna())
    test_groups = set(test_df[GROUP].dropna())

    overlap = train_groups.intersection(test_groups)

    print(f"\nUnique training host stars: {len(train_groups)}")
    print(f"Unique test host stars:     {len(test_groups)}")
    print(f"Shared host stars:          {len(overlap)}")

    if overlap:
        print("\nWARNING:")
        print("The existing train/test split contains shared host stars.")
        print("Grouped validation will still protect CV folds,")
        print("but the final test split is not fully host-independent.")

    # ------------------------------------------------------------
    # Prepare arrays.
    # ------------------------------------------------------------

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET].astype(int)
    groups = train_df[GROUP]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET].astype(int)

    # ------------------------------------------------------------
    # Grouped cross-validation.
    # ------------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print("\nChecking grouped CV folds...")

    for fold, (tr_idx, val_idx) in enumerate(
        cv.split(X_train, y_train, groups),
        start=1
    ):
        train_hosts = set(groups.iloc[tr_idx].dropna())
        val_hosts = set(groups.iloc[val_idx].dropna())

        shared = train_hosts.intersection(val_hosts)

        print(
            f"  Fold {fold}: "
            f"train={len(tr_idx)}, "
            f"validation={len(val_idx)}, "
            f"shared hosts={len(shared)}"
        )

        if shared:
            raise RuntimeError(
                f"Host leakage detected in fold {fold}"
            )

    # ------------------------------------------------------------
    # Models.
    # ------------------------------------------------------------

    models = {
        "random_forest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                )
            ),
        ]),

        "gradient_boosting": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                )
            ),
        ]),
    }

    OUT_DIR.mkdir(exist_ok=True)

    results = []

    for name, model in models.items():

        print("\n" + "=" * 78)
        print(f"MODEL: {name}")
        print("=" * 78)

        print("\nRunning 5-fold StratifiedGroupKFold...")

        cv_prob = cross_val_predict(
            model,
            X_train,
            y_train,
            groups=groups,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        cv_metrics = metrics(
            y_train,
            cv_prob
        )

        print_metrics(
            "GROUPED CROSS-VALIDATION:",
            cv_metrics
        )

        # --------------------------------------------------------
        # Final training on all training data.
        # --------------------------------------------------------

        print("\nFitting final model on complete training set...")

        model.fit(X_train, y_train)

        test_prob = model.predict_proba(X_test)[:, 1]

        test_metrics = metrics(
            y_test,
            test_prob
        )

        print_metrics(
            "HOST-INDEPENDENT TEST SET:",
            test_metrics
        )

        # --------------------------------------------------------
        # Save predictions.
        # --------------------------------------------------------

        predictions = test_df[
            [GROUP, TARGET]
        ].copy()

        predictions["predicted_probability"] = test_prob
        predictions["predicted_class"] = (
            test_prob >= 0.5
        ).astype(int)

        prediction_path = (
            OUT_DIR /
            f"{name}_test_predictions.csv"
        )

        predictions.to_csv(
            prediction_path,
            index=False
        )

        # --------------------------------------------------------
        # Save model.
        # --------------------------------------------------------

        import joblib

        model_path = (
            OUT_DIR /
            f"{name}.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        row = {
            "model": name,
            "n_features": len(feature_cols),
        }

        row.update({
            f"cv_{k}": v
            for k, v in cv_metrics.items()
        })

        row.update({
            f"test_{k}": v
            for k, v in test_metrics.items()
        })

        results.append(row)

    # ------------------------------------------------------------
    # Comparison table.
    # ------------------------------------------------------------

    comparison = pd.DataFrame(results)

    comparison_path = (
        OUT_DIR /
        "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    # ------------------------------------------------------------
    # Metadata.
    # ------------------------------------------------------------

    metadata = {
        "dataset": str(DATA_PATH),
        "n_rows": len(df),
        "n_training_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_physics_features": len(feature_cols),
        "features": feature_cols,
        "target": TARGET,
        "group": GROUP,
        "cv": "StratifiedGroupKFold",
        "cv_folds": 5,
        "random_state": 42,
        "test_host_overlap": len(overlap),
    }

    with open(
        OUT_DIR / "run_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print("\n" + "=" * 78)
    print("MODEL COMPARISON")
    print("=" * 78)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nArtifacts written to:")
    print(OUT_DIR.resolve())

    print("=" * 78)


if __name__ == "__main__":
    main()