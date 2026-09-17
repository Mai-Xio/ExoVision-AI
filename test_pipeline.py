"""
ExoVision AI - Final Project Test Suite

Tests the actual final project implementation:
- dataset integrity
- final feature list
- leakage protection
- grouped train/test split
- trained model artifacts
- candidate screening
- explainability artifacts
- FastAPI endpoints
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "koi_training_data.csv"
CANDIDATES_PATH = ROOT / "artifacts_final" / "final_earthlike_candidates.csv"
FEATURE_LIST_PATH = ROOT / "artifacts_final" / "feature_list.json"
RF_PATH = ROOT / "artifacts_final" / "random_forest_final.joblib"
GB_PATH = ROOT / "artifacts_final" / "gradient_boosting_final.joblib"
MODEL_COMPARISON_PATH = ROOT / "artifacts_final" / "model_comparison.csv"
FAMILY_IMPORTANCE_PATH = (
    ROOT / "artifacts_final" / "final_feature_family_importance.csv"
)


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def feature_list():
    assert FEATURE_LIST_PATH.exists(), (
        f"Feature list not found: {FEATURE_LIST_PATH}"
    )

    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    return data["features"]


@pytest.fixture(scope="module")
def candidates():
    assert CANDIDATES_PATH.exists(), (
        f"Candidate file not found: {CANDIDATES_PATH}"
    )
    return pd.read_csv(CANDIDATES_PATH)


# ---------------------------------------------------------------------------
# DATA INTEGRITY
# ---------------------------------------------------------------------------

def test_training_dataset_exists(dataset):
    assert len(dataset) > 0


def test_training_dataset_contains_binary_target(dataset):
    assert "target_binary" in dataset.columns
    assert set(dataset["target_binary"].dropna().unique()).issubset({0, 1})


def test_training_dataset_contains_both_classes(dataset):
    classes = set(dataset["target_binary"].dropna().unique())
    assert classes == {0, 1}


def test_training_dataset_contains_host_identifier(dataset):
    assert "kepid" in dataset.columns
    assert dataset["kepid"].notna().all()


def test_train_test_split_exists(dataset):
    assert "split" in dataset.columns
    assert set(dataset["split"].dropna().unique()) == {"train", "test"}


def test_train_test_host_groups_do_not_overlap(dataset):
    train_hosts = set(
        dataset.loc[dataset["split"] == "train", "kepid"].dropna()
    )
    test_hosts = set(
        dataset.loc[dataset["split"] == "test", "kepid"].dropna()
    )

    assert train_hosts
    assert test_hosts
    assert not (train_hosts & test_hosts)


# ---------------------------------------------------------------------------
# FINAL FEATURE SET
# ---------------------------------------------------------------------------

def test_final_feature_list_has_expected_size(feature_list):
    assert len(feature_list) == 35


def test_all_final_features_exist_in_dataset(dataset, feature_list):
    missing = [feature for feature in feature_list if feature not in dataset.columns]
    assert not missing, f"Missing final features: {missing}"


def test_final_features_are_physics_engineered_features(feature_list):
    non_physics = [feature for feature in feature_list
                   if not feature.startswith("f_")]
    assert not non_physics, (
        f"Non physics-engineered features found: {non_physics}"
    )


def test_leakage_columns_are_not_in_final_features(feature_list):
    quarantined = {
        "koi_score",
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
    }

    overlap = set(feature_list) & quarantined

    assert not overlap, (
        f"Leakage-prone columns found in final features: {sorted(overlap)}"
    )


def test_zero_variance_features_are_excluded(feature_list):
    excluded = {
        "f_insol_ratio",
        "f_teq_ratio",
        "f_period_frac_err",
    }

    overlap = set(feature_list) & excluded

    assert not overlap, (
        f"Zero-variance features found in final feature list: {sorted(overlap)}"
    )


def test_target_is_not_a_feature(feature_list):
    assert "target_binary" not in feature_list
    assert "split" not in feature_list
    assert "kepid" not in feature_list


# ---------------------------------------------------------------------------
# MODEL ARTIFACTS
# ---------------------------------------------------------------------------

def test_random_forest_artifact_exists():
    assert RF_PATH.exists()


def test_gradient_boosting_artifact_exists():
    assert GB_PATH.exists()


def test_random_forest_can_be_loaded():
    model = joblib.load(RF_PATH)
    assert model is not None


def test_gradient_boosting_can_be_loaded():
    model = joblib.load(GB_PATH)
    assert model is not None


def test_random_forest_accepts_35_features(dataset, feature_list):
    model = joblib.load(RF_PATH)

    X = dataset[feature_list].copy().head(5)

    predictions = model.predict_proba(X)

    assert predictions.shape == (5, 2)
    assert np.isfinite(predictions).all()
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)


def test_random_forest_probability_sums_to_one(dataset, feature_list):
    model = joblib.load(RF_PATH)

    X = dataset[feature_list].copy().head(10)

    probabilities = model.predict_proba(X)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(10),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# MODEL RESULTS
# ---------------------------------------------------------------------------

def test_model_comparison_exists():
    assert MODEL_COMPARISON_PATH.exists()


def test_model_comparison_contains_both_models():
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)

    text = comparison.to_string().lower()

    assert "random" in text
    assert "gradient" in text


# ---------------------------------------------------------------------------
# EARTH-LIKE CANDIDATES
# ---------------------------------------------------------------------------

def test_candidate_artifact_exists(candidates):
    assert len(candidates) > 0


def test_final_candidate_count(candidates):
    assert len(candidates) == 10


def test_candidate_probabilities_are_valid(candidates):
    assert "planet_probability" in candidates.columns

    probabilities = candidates["planet_probability"].dropna()

    assert len(probabilities) == 10
    assert probabilities.between(0, 1).all()


def test_candidates_have_earth_like_radius(candidates):
    assert "koi_prad" in candidates.columns

    assert candidates["koi_prad"].between(0.8, 1.25).all()


def test_candidates_have_earth_like_insolation(candidates):
    assert "koi_insol" in candidates.columns

    assert candidates["koi_insol"].between(0.5, 2.0).all()


def test_candidates_meet_probability_threshold(candidates):
    assert (
        candidates["planet_probability"] >= 0.80
    ).all()


def test_candidate_scores_are_valid(candidates):
    assert "earthlike_score" in candidates.columns

    scores = candidates["earthlike_score"].dropna()

    assert len(scores) == 10
    assert scores.between(0, 1).all()


def test_physics_consistency_fields_exist(candidates):
    required = {
        "density_consistent",
        "duration_consistent",
        "depth_consistent",
        "overall_physics_consistent",
    }

    missing = required - set(candidates.columns)

    assert not missing, f"Missing physics fields: {sorted(missing)}"


# ---------------------------------------------------------------------------
# EXPLAINABILITY
# ---------------------------------------------------------------------------

def test_feature_family_importance_exists():
    assert FAMILY_IMPORTANCE_PATH.exists()


def test_feature_family_importance_is_not_empty():
    df = pd.read_csv(FAMILY_IMPORTANCE_PATH)

    assert len(df) > 0


# ---------------------------------------------------------------------------
# FASTAPI
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "ExoVision AI API"
    assert body["status"] == "online"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["feature_count"] == 35


def test_model_endpoint(client):
    response = client.get("/api/model")

    assert response.status_code == 200

    body = response.json()

    assert body["feature_count"] == 35


def test_model_performance_endpoint(client):
    response = client.get("/api/model/performance")

    assert response.status_code == 200


def test_candidates_endpoint(client):
    response = client.get("/api/candidates")

    assert response.status_code == 200

    body = response.json()

    assert "candidates" in body
    assert len(body["candidates"]) == 10


def test_candidate_lookup_endpoint(client):
    response = client.get("/api/candidates/K02717.01")

    assert response.status_code == 200

    body = response.json()

    assert body["kepoi_name"] == "K02717.01"


def test_unknown_candidate_returns_404(client):
    response = client.get("/api/candidates/DOES_NOT_EXIST.99")

    assert response.status_code == 404


def test_candidate_screening_endpoint(client):
    response = client.post(
        "/api/candidates/K02717.01/screen"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["kepoi_name"] == "K02717.01"
    assert 0 <= body["planet_probability"] <= 1
    assert isinstance(body["passes_screen"], bool)
    assert "screening_criteria" in body
    assert "physics_consistency" in body
    assert "scores" in body
    assert "physical_values" in body


def test_candidate_screening_returns_valid_score(client):
    response = client.post(
        "/api/candidates/K02717.01/screen"
    )

    body = response.json()

    score = body["scores"]["earthlike_screening_score"]

    assert 0 <= score <= 1


def test_explainability_endpoint(client):
    response = client.get(
        "/api/explainability/families"
    )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# FINAL SCIENTIFIC SANITY CHECK
# ---------------------------------------------------------------------------

def test_top_candidate_has_high_probability(candidates):
    top = candidates.iloc[0]

    assert top["planet_probability"] >= 0.80


def test_top_candidate_has_physics_consistency(candidates):
    top = candidates.iloc[0]

    assert bool(top["overall_physics_consistent"]) is True