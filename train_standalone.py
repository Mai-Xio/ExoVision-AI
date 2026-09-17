import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict


DATA_FILE = Path("koi_training_data.csv")
OUTPUT_DIR = Path("artifacts_standalone")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("=" * 78)
print("EXOVISION AI — STANDALONE TRAINING")
print("=" * 78)
print(f"Dataset: {DATA_FILE}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print()


# ---------------------------------------------------------
# 2. DEFINE TARGET
# ---------------------------------------------------------

TARGET = "target_binary"

if TARGET not in df.columns:
    raise RuntimeError(f"Missing target column: {TARGET}")


# ---------------------------------------------------------
# 3. USE THE EXISTING TRAIN/TEST SPLIT
# ---------------------------------------------------------

if "split" not in df.columns:
    raise RuntimeError("Dataset does not contain the required 'split' column.")

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

print(f"Training rows: {len(train_df)}")
print(f"Test rows:     {len(test_df)}")
print()


# ---------------------------------------------------------
# 4. REMOVE LEAKAGE / NON-FEATURE COLUMNS
# ---------------------------------------------------------

# These are deliberately excluded from the scientifically
# valid model.

EXCLUDED = {
    TARGET,
    "split",
    "koi_score",
}

# Quarantined false-positive disposition flags.
EXCLUDED.update(
    c for c in df.columns
    if c.startswith("koi_fpflag_")
)

# Identification / categorical / raw metadata fields that
# should not automatically become ML features.
IDENTIFIER_COLUMNS = {
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_vet_stat",
    "koi_tce_delivname",
}

EXCLUDED.update(
    c for c in IDENTIFIER_COLUMNS
    if c in df.columns
)


# Keep only numeric columns.

candidate_features = [
    c for c in train_df.columns
    if c not in EXCLUDED
    and pd.api.types.is_numeric_dtype(train_df[c])
]


if not candidate_features:
    raise RuntimeError("No usable numeric features were found.")


print(f"Using {len(candidate_features)} features.")
print()

print("Excluded leakage/control columns:")
for c in sorted(EXCLUDED):
    if c in df.columns:
        print(f"  - {c}")

print()


X_train = train_df[candidate_features]
y_train = train_df[TARGET].astype(int)

X_test = test_df[candidate_features]
y_test = test_df[TARGET].astype(int)


# ---------------------------------------------------------
# 5. MODELS
# ---------------------------------------------------------

models = {
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}


# ---------------------------------------------------------
# 6. METRICS
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_proba),
    }


# ---------------------------------------------------------
# 7. TRAIN + CROSS-VALIDATION + TEST
# ---------------------------------------------------------

all_results = []

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


for model_name, model in models.items():

    print("=" * 78)
    print(f"MODEL: {model_name}")
    print("=" * 78)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    print("Running 5-fold cross-validation...")

    oof_proba = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    oof_pred = (oof_proba >= 0.5).astype(int)

    cv_metrics = calculate_metrics(
        y_train,
        oof_pred,
        oof_proba,
    )

    print("\nCross-validation:")
    for key, value in cv_metrics.items():
        print(f"  {key:20s}: {value:.4f}")


    print("\nFitting final model on complete training set...")

    pipeline.fit(X_train, y_train)

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_pred,
        test_proba,
    )

    print("\nTEST SET:")
    for key, value in test_metrics.items():
        print(f"  {key:20s}: {value:.4f}")


    # Save model
    model_path = OUTPUT_DIR / f"{model_name}.joblib"

    joblib.dump(
        {
            "pipeline": pipeline,
            "features": candidate_features,
            "model_name": model_name,
            "scientifically_valid": True,
        },
        model_path,
    )


    # Save predictions
    predictions = pd.DataFrame({
        "row_index": test_df.index,
        "y_true": y_test.to_numpy(),
        "y_proba": test_proba,
        "y_pred": test_pred,
    })

    predictions.to_csv(
        OUTPUT_DIR / f"{model_name}_test_predictions.csv",
        index=False,
    )


    all_results.append({
        "model": model_name,
        "n_features": len(candidate_features),
        **{
            f"cv_{k}": v
            for k, v in cv_metrics.items()
        },
        **{
            f"test_{k}": v
            for k, v in test_metrics.items()
        },
    })


# ---------------------------------------------------------
# 8. SAVE COMPARISON
# ---------------------------------------------------------

comparison = pd.DataFrame(all_results)

comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

print()
print("=" * 78)
print("MODEL COMPARISON")
print("=" * 78)
print(comparison.round(4).to_string(index=False))

print()
print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")
print("=" * 78)