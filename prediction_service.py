"""
Prediction service.

Two invariants enforced here rather than trusted:

1. **Models load once, at application startup.** A request never triggers a
   `joblib.load`, and nothing in the API can retrain. If an API can retrain a
   model, you cannot say which model produced a given result.

2. **The leaky control can never be served.** `load()` refuses any artifact
   whose metadata says `scientifically_valid: false`, and `predict()` re-checks.
   Belt and braces, because the failure mode is silent: a leaky model returns
   confident, plausible predictions that happen to be worthless.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This classification is a machine-learning triage of catalogue parameters. "
    "It does not confirm a planet and does not replace astrophysical validation. "
    "A Kepler Object of Interest labelled CANDIDATE has not been refuted, which "
    "is not the same as having been confirmed."
)


class ModelNotLoadedError(RuntimeError):
    pass


class LeakyModelRefused(RuntimeError):
    """Raised when something tries to serve the leaky control."""


class PredictionService:
    """Holds the primary model for the process lifetime."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.pipeline: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.primary: dict[str, Any] = {}
        self.feature_columns: list[str] = []
        self.available: list[dict[str, Any]] = []
        self.data_source_banner: str = ""

    # -- startup ---------------------------------------------------------
    def load(self) -> None:
        primary_path = self.settings.metrics_dir / "primary_model.json"
        if not primary_path.exists():
            logger.warning(
                "no primary_model.json; prediction endpoints will return 503. "
                "Run training and then ml.evaluation.compare_models.")
            return

        self.primary = json.loads(primary_path.read_text(encoding="utf-8"))
        tag = self.primary["tag"]
        model_path = self.settings.models_dir / f"{tag}.joblib"
        if not model_path.exists():
            logger.warning("primary artifact %s missing", model_path.name)
            return

        bundle = joblib.load(model_path)
        metadata = bundle.get("metadata", {})
        if not metadata.get("scientifically_valid", True):
            raise LeakyModelRefused(
                f"{tag} is a leaky control and must never be served. "
                "Re-run ml.evaluation.compare_models to select a valid primary model.")

        self.pipeline = bundle["pipeline"]
        self.metadata = metadata
        self.feature_columns = list(metadata.get("features", []))
        self.data_source_banner = self.primary.get("data_source_banner", "")
        logger.info("loaded primary model %s (%d features)", tag, len(self.feature_columns))

        self._index_available_models()

    def _index_available_models(self) -> None:
        """Catalogue every trained model's metadata for GET /api/models."""
        self.available = []
        for path in sorted(self.settings.metrics_dir.glob("*.json")):
            if path.name in {"primary_model.json", "leakage_experiment.json"}:
                continue
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if "model_name" not in meta:
                continue
            self.available.append({
                "tag": f"{meta['feature_set']}__{meta['model_name']}",
                "model_name": meta["model_name"],
                "feature_set": meta["feature_set"],
                "scientifically_valid": meta.get("scientifically_valid", True),
                "validity_note": meta.get("validity_note", ""),
                "n_features": meta.get("n_features"),
                "training_date": meta.get("training_date"),
                "random_seed": meta.get("random_seed"),
                "cv_strategy": meta.get("cv_strategy"),
                "metrics": meta.get("metrics", {}),
                "is_primary": f"{meta['feature_set']}__{meta['model_name']}"
                              == self.primary.get("tag"),
            })

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    # -- inference -------------------------------------------------------
    def _frame_from_request(self, payload: dict[str, Any]) -> pd.DataFrame:
        """
        Build a one-row frame carrying every column the pipeline expects.

        Missing inputs become NaN and are handled by the imputer inside the
        pipeline — the same imputer, fitted on the same training folds, that
        was used during evaluation. Reconstructing that here would be a
        train/serve skew bug.
        """
        from ml.preprocessing.features import engineer_features

        row = {k: v for k, v in payload.items() if v is not None}
        df = pd.DataFrame([row])

        # Derive the engineered features from the supplied catalogue values,
        # using exactly the training-time code path.
        df = engineer_features(df)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        return df[self.feature_columns]

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.ready:
            raise ModelNotLoadedError(
                "No model is loaded. The service was started without trained artifacts.")
        if not self.metadata.get("scientifically_valid", True):
            raise LeakyModelRefused("refusing to serve a leaky control model")

        X = self._frame_from_request(payload)
        classes = list(self.pipeline.classes_)
        proba = float(self.pipeline.predict_proba(X)[0][classes.index(1)])

        consistency = {
            "density_ratio": _maybe(X, "f_log_rho_ratio"),
            "duration_ratio": _maybe(X, "f_duration_ratio"),
            "depth_ratio": _maybe(X, "f_depth_ratio"),
        }
        return {
            "prediction": "PLANET CANDIDATE" if proba >= 0.5 else "FALSE POSITIVE",
            "probability": round(proba, 4),
            "model": self.metadata.get("model_name", "unknown"),
            "feature_set": self.metadata.get("feature_set", "unknown"),
            "scientifically_valid": True,
            "physics_consistency": consistency,
            "disclaimer": DISCLAIMER,
        }

    def explain(self, payload: dict[str, Any], top_n: int = 8) -> dict[str, Any]:
        if not self.ready:
            raise ModelNotLoadedError("No model is loaded.")
        from ml.explainability.shap_analysis import HAS_SHAP, explain_one

        if not HAS_SHAP:
            raise RuntimeError("shap is not installed in this environment")
        X = self._frame_from_request(payload)
        out = explain_one(self.pipeline, X, top_n=top_n)
        total = out.get("total_absolute_contribution") or 0.0
        out["physics_feature_share"] = (
            round(out["physics_feature_contribution"] / total, 4) if total else None)
        return out

    def model_catalogue(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "models": self.available,
            "data_source_banner": self.data_source_banner,
            "note": ("Models marked scientifically_valid=false are leaky controls. "
                     "They are retained to demonstrate target leakage and are never "
                     "used for prediction, ranking or deployment."),
        }


def _maybe(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    v = df[col].iloc[0]
    return round(float(v), 4) if pd.notna(v) and np.isfinite(v) else None


#: Process-wide singleton, populated by the FastAPI lifespan handler.
prediction_service = PredictionService()
