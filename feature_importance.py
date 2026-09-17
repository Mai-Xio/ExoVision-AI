import pandas as pd
import joblib

from pathlib import Path

DATASET = Path("koi_training_data.csv")
ARTIFACTS = Path("artifacts_physics_grouped")

df = pd.read_csv(DATASET)

FEATURES = [c for c in df.columns if c.startswith("f_")]

print("=" * 78)
print("EXOVISION AI — PHYSICS FEATURE IMPORTANCE")
print("=" * 78)

print(f"Physics features: {len(FEATURES)}")

for model_name in ["random_forest", "gradient_boosting"]:

    model_path = ARTIFACTS / f"{model_name}.joblib"

    pipeline = joblib.load(model_path)

    # The trained estimator is inside the sklearn Pipeline.
    # Usually the final step is the classifier.
    estimator = pipeline.steps[-1][1]

    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError(
            f"{model_name}: final pipeline step "
            f"{type(estimator).__name__} does not provide "
            f"feature_importances_."
        )

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": estimator.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance["rank"] = importance.index + 1

    importance = importance[
        ["rank", "feature", "importance"]
    ]

    output_file = (
        ARTIFACTS /
        f"{model_name}_feature_importance.csv"
    )

    importance.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 78)
    print(model_name.upper())
    print("=" * 78)

    print()
    print("TOP 15 FEATURES")
    print("-" * 78)

    print(
        importance.head(15).to_string(index=False)
    )

    print()
    print("Saved:")
    print(output_file)

print()
print("=" * 78)
print("Feature importance analysis complete.")
print("=" * 78)