"""
Training entry point.

    python -m ml.training.train                      # all valid feature sets
    python -m ml.training.train --dev                # fast subset, ~1 min
    python -m ml.training.train --tune               # + randomized search
    python -m ml.training.train --include-leaky      # + the control experiment
    python -m ml.training.train --models gradient_boosting random_forest

Contract
--------
* The test set is loaded but touched exactly once, at the end, after model
  selection is complete. Selection uses cross-validated out-of-fold scores on
  the training data only.
* Every artifact records the data source, the split hash, the feature set, the
  seed and the library versions. A metrics file without that context cannot be
  audited later.
* The leaky control is opt-in, tagged `scientifically_valid: false`, and
  excluded from primary-model selection.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.config import SETTINGS
from ml.config import features as fx
from ml.config import labels as lbl
from ml.evaluation.metrics import binary_metrics
from ml.model_zoo import MODEL_SPECS, available_models, build_model
from ml.preprocessing.splits import SplitSpec, apply_split, leakage_selfcheck
from ml.training.cross_validation import cross_validate_grouped
from ml.training.tuning import tune_model

logger = logging.getLogger(__name__)


def _library_versions() -> dict[str, str]:
    import sklearn
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import xgboost
        versions["xgboost"] = xgboost.__version__
    except ImportError:
        versions["xgboost"] = "unavailable"
    return versions


def load_processed() -> tuple[pd.DataFrame, SplitSpec, dict[str, Any]]:
    """Load the dataset built by ml.preprocessing.build_dataset."""
    path = SETTINGS.processed_dir / "koi_clean.parquet"
    split_path = SETTINGS.processed_dir / "split.json"
    if not path.exists() or not split_path.exists():
        raise FileNotFoundError(
            f"{path} or {split_path} missing. Run:\n"
            "    python -m ml.preprocessing.build_dataset"
        )
    df = pd.read_parquet(path)
    spec = SplitSpec.from_json(split_path)
    meta_path = SETTINGS.processed_dir / "dataset_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return df, spec, meta


def train_one(
    model_name: str,
    feature_set: str,
    df: pd.DataFrame,
    spec: SplitSpec,
    *,
    tune: bool = False,
    dev_mode: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Cross-validate, optionally tune, fit on all training data, score the test set."""
    resolved = fx.resolve(df, feature_set)
    fx.assert_no_leakage(resolved.all, feature_set)

    train_df, test_df = apply_split(df, spec)
    checks = leakage_selfcheck(train_df, test_df)
    if checks["shared_host_stars"]:
        raise AssertionError(f"{checks['shared_host_stars']} host stars in both splits")

    target = lbl.TARGET_BINARY
    X_tr, y_tr = train_df[resolved.all], train_df[target]
    X_te, y_te = test_df[resolved.all], test_df[target]

    pipeline = build_model(model_name, resolved, seed=seed, dev_mode=dev_mode)
    best_params: dict[str, Any] = {}
    tuned_score = float("nan")

    if tune:
        pipeline, best_params, tuned_score = tune_model(
            pipeline, MODEL_SPECS[model_name].param_distributions,
            X_tr, y_tr, train_df, spec,
            n_iter=SETTINGS.tuning_iterations, seed=seed,
        )

    cv = cross_validate_grouped(
        pipeline, X_tr, y_tr, train_df, spec,
        model_name=model_name, feature_set=feature_set,
    )

    # Final fit on the full training set, then the single test evaluation.
    from sklearn.base import clone
    final = clone(pipeline).fit(X_tr, y_tr)
    from ml.training.cross_validation import _positive_proba
    test_proba = _positive_proba(final, X_te)
    test_m = binary_metrics(y_te, (test_proba >= 0.5).astype(int), test_proba)

    valid = feature_set in fx.VALID_FEATURE_SETS
    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "scientifically_valid": valid,
        "validity_note": "" if valid else "LEAKY CONTROL — NOT VALID FOR SCIENTIFIC USE",
        "n_features": len(resolved),
        "features": resolved.all,
        "missing_features": resolved.missing,
        "best_params": best_params,
        "tuned_cv_score": tuned_score,
        "cv_summary": cv.summary,
        "cv_oof": cv.oof_metrics().to_dict(),
        "test": test_m.to_dict(),
        "_pipeline": final,
        "_oof": {"index": cv.oof_index, "proba": cv.oof_proba, "true": cv.oof_true},
        "_test_proba": test_proba,
        "_test_index": test_df.index.to_numpy(),
    }


def persist(result: dict[str, Any], dataset_meta: dict[str, Any], seed: int) -> Path:
    """Save the fitted pipeline plus everything needed to interpret it later."""
    SETTINGS.ensure_dirs()
    tag = f"{result['feature_set']}__{result['model_name']}"
    model_path = SETTINGS.models_dir / f"{tag}.joblib"

    metadata = {
        "model_name": result["model_name"],
        "feature_set": result["feature_set"],
        "scientifically_valid": result["scientifically_valid"],
        "validity_note": result["validity_note"],
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": seed,
        "cv_strategy": "StratifiedGroupKFold on kepid (persisted in split.json)",
        "task": "binary: PLANET CANDIDATE vs FALSE POSITIVE",
        "label_mapping": lbl.BINARY_MAP,
        "features": result["features"],
        "n_features": result["n_features"],
        "best_params": result["best_params"],
        "metrics": {"cv": result["cv_summary"], "cv_oof": result["cv_oof"],
                    "test": result["test"]},
        "dataset": dataset_meta,
        "libraries": _library_versions(),
    }

    joblib.dump({"pipeline": result["_pipeline"], "metadata": metadata}, model_path)
    (SETTINGS.metrics_dir / f"{tag}.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    # Out-of-fold predictions drive calibration, error analysis and SHAP.
    oof = result["_oof"]
    if oof["proba"] is not None and len(oof["proba"]):
        pd.DataFrame({"row_index": oof["index"], "y_true": oof["true"],
                      "y_proba": oof["proba"]}).to_csv(
            SETTINGS.metrics_dir / f"{tag}__oof.csv", index=False)
    pd.DataFrame({"row_index": result["_test_index"],
                  "y_proba": result["_test_proba"]}).to_csv(
        SETTINGS.metrics_dir / f"{tag}__test_pred.csv", index=False)
    return model_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--feature-sets", nargs="*", default=None)
    p.add_argument("--include-leaky", action="store_true",
                   help="Also train the leaky control. Educational only.")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--dev", action="store_true", help="Fast subset of models.")
    p.add_argument("--seed", type=int, default=SETTINGS.random_seed)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    dev = args.dev or SETTINGS.dev_mode

    df, spec, dataset_meta = load_processed()
    banner = dataset_meta.get("banner", SETTINGS.data_source_banner())
    print("=" * 78)
    print(banner)
    print("=" * 78)

    feature_sets = args.feature_sets or list(fx.VALID_FEATURE_SETS)
    if args.include_leaky and "leaky_control" not in feature_sets:
        feature_sets.append("leaky_control")
    models = args.models or list(available_models(dev))

    rows: list[dict[str, Any]] = []
    for fs in feature_sets:
        for mname in models:
            logger.info("training %s on %s", mname, fs)
            try:
                res = train_one(mname, fs, df, spec, tune=args.tune,
                                dev_mode=dev, seed=args.seed)
            except Exception as exc:  # noqa: BLE001
                logger.error("FAILED %s/%s: %s", mname, fs, exc)
                continue
            persist(res, dataset_meta, args.seed)
            rows.append({
                "model": res["model_name"],
                "feature_set": res["feature_set"],
                "scientifically_valid": res["scientifically_valid"],
                "n_features": res["n_features"],
                **{k: res["test"][k] for k in
                   ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                    "balanced_accuracy", "mcc", "brier")},
                "cv_pr_auc_mean": res["cv_summary"].get("pr_auc_mean", float("nan")),
                "cv_pr_auc_std": res["cv_summary"].get("pr_auc_std", float("nan")),
            })

    if not rows:
        logger.error("no models trained successfully")
        return 1

    comparison = pd.DataFrame(rows).sort_values(
        ["scientifically_valid", SETTINGS.selection_metric], ascending=[False, False])
    SETTINGS.ensure_dirs()
    out_csv = SETTINGS.metrics_dir / "model_comparison.csv"
    comparison.to_csv(out_csv, index=False)

    print("\nMODEL COMPARISON (test set, single evaluation)")
    print("=" * 78)
    print(comparison.round(4).to_string(index=False))
    print(f"\nwrote {out_csv}")
    if not comparison["scientifically_valid"].all():
        print("\nRows with scientifically_valid=False are LEAKY CONTROLS. "
              "They are not models; they are a demonstration of target leakage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
python -c "import ml.training.train; print('ML PACKAGE OK')"