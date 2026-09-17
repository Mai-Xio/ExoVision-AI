import pandas as pd
import numpy as np
import joblib

from pathlib import Path
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
    confusion_matrix,
)

DATASET = Path("koi_training_data.csv")
ARTIFACTS = Path("artifacts_physics_grouped")

df = pd.read_csv(DATASET)

FEATURES = [c for c in df.columns if c.startswith("f_")]

train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()

X_test = test[FEATURES]
y_test = test["target_binary"]

print("=" * 78)
print("EXOVISION AI — MODEL EVALUATION")
print("=" * 78)

for model_name in ["random_forest", "gradient_boosting"]:

    model_path = ARTIFACTS / f"{model_name}.joblib"

    model = joblib.load(model_path)

    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction),
        "recall": recall_score(y_test, prediction),
        "f1": f1_score(y_test, prediction),
        "roc_auc": roc_auc_score(y_test, probability),
        "pr_auc": average_precision_score(y_test, probability),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "mcc": matthews_corrcoef(y_test, prediction),
        "brier": brier_score_loss(y_test, probability),
    }

    print()
    print("=" * 78)
    print(model_name.upper())
    print("=" * 78)

    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")

    cm = confusion_matrix(y_test, prediction)

    print()
    print("CONFUSION MATRIX")
    print(cm)

    output = test[
        [
            "kepoi_name",
            "kepid",
            "target_binary",
            "koi_disposition",
        ]
    ].copy()

    output["predicted_probability"] = probability
    output["predicted_class"] = prediction

    output.to_csv(
        ARTIFACTS / f"{model_name}_evaluation_predictions.csv",
        index=False,
    )

print()
print("=" * 78)
print("Evaluation complete.")
print("=" * 78)