from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = ROOT_DIR / "artifacts_final"

MODEL_PATH = ARTIFACTS_DIR / "random_forest_final.joblib"
FEATURE_LIST_PATH = ARTIFACTS_DIR / "feature_list.json"
CANDIDATES_PATH = ARTIFACTS_DIR / "final_earthlike_candidates.csv"
TRAINING_DATA_PATH = ROOT_DIR / "koi_training_data.csv"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"
FAMILY_IMPORTANCE_PATH = (
    ARTIFACTS_DIR / "final_feature_family_importance.csv"
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Could not load model from {MODEL_PATH}: {exc}"
    )

# ============================================================
# LOAD TRAINING DATA
# ============================================================

try:
    training_df = pd.read_csv(TRAINING_DATA_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Could not load training data from "
        f"{TRAINING_DATA_PATH}: {exc}"
    )

try:
    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as file:
        feature_list = json.load(file)

    if isinstance(feature_list, list):
        FEATURES = feature_list
    else:
        FEATURES = feature_list["features"]

except Exception as exc:
    raise RuntimeError(
        f"Could not load feature list from "
        f"{FEATURE_LIST_PATH}: {exc}"
    )


# ============================================================
# LOAD CANDIDATES
# ============================================================

try:
    candidates_df = pd.read_csv(CANDIDATES_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Could not load candidates from "
        f"{CANDIDATES_PATH}: {exc}"
    )


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

try:
    model_comparison_df = pd.read_csv(
        MODEL_COMPARISON_PATH
    )
except Exception:
    model_comparison_df = pd.DataFrame()


# ============================================================
# LOAD FEATURE FAMILY IMPORTANCE
# ============================================================

try:
    family_importance_df = pd.read_csv(
        FAMILY_IMPORTANCE_PATH
    )
except Exception:
    family_importance_df = pd.DataFrame()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ExoVision AI API",
    description=(
        "API for AI-assisted screening of Kepler "
        "exoplanet candidates using physics-engineered "
        "features."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):
    features: dict[str, float | None] = Field(
        ...,
        description=(
            "Dictionary containing the 35 features "
            "used by the final model."
        ),
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "name": "ExoVision AI API",
        "version": "1.0.0",
        "status": "online",
        "model": "Random Forest",
        "feature_count": len(FEATURES),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(FEATURES),
        "candidate_count": len(candidates_df),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/api/model")
def model_info():
    return {
        "model": "Random Forest",
        "feature_count": len(FEATURES),
        "features": FEATURES,
    }


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.get("/api/model/performance")
def model_performance():
    if model_comparison_df.empty:
        return {
            "status": "unavailable",
            "models": [],
        }

    return {
        "models": model_comparison_df
        .replace({np.nan: None})
        .to_dict(orient="records")
    }


# ============================================================
# FEATURE IMPORTANCE BY FAMILY
# ============================================================

@app.get("/api/explainability/families")
def feature_family_importance():
    if family_importance_df.empty:
        return {
            "status": "unavailable",
            "families": [],
        }

    return {
        "families": family_importance_df
        .replace({np.nan: None})
        .to_dict(orient="records")
    }


# ============================================================
# CANDIDATE LIST
# ============================================================

@app.get("/api/candidates")
def get_candidates():
    data = (
        candidates_df
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    return {
        "count": len(data),
        "candidates": data,
    }


# ============================================================
# SINGLE CANDIDATE
# ============================================================

@app.get("/api/candidates/{kepoi_name}")
def get_candidate(kepoi_name: str):

    result = candidates_df[
        candidates_df["kepoi_name"].astype(str)
        == kepoi_name
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate {kepoi_name} not found.",
        )

    record = (
        result.iloc[0]
        .replace({np.nan: None})
        .to_dict()
    )

    return record


# ============================================================
# PREDICTION
# ============================================================

@app.post("/api/predict")
def predict(request: PredictionRequest):

    supplied = request.features

    missing = [
        feature
        for feature in FEATURES
        if feature not in supplied
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features.",
                "missing_features": missing,
            },
        )

    values = {}

    for feature in FEATURES:
        value = supplied[feature]

        if value is None:
            values[feature] = np.nan
        else:
            try:
                values[feature] = float(value)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Feature '{feature}' must be numeric "
                        "or null."
                    ),
                )

    input_df = pd.DataFrame(
        [values],
        columns=FEATURES,
    )

    try:
        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        prediction = int(
            model.predict(input_df)[0]
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )

    return {
        "prediction": prediction,
        "planet_probability": probability,
        "model": "Random Forest",
        "feature_count": len(FEATURES),
    }
@app.post("/api/candidates/{kepoi_name}/screen")
def screen_candidate(kepoi_name: str):
    """
    Screen an existing KOI candidate directly from the training dataset.

    The frontend only needs to provide the KOI name.
    The backend retrieves the required features and runs the
    complete model + Earth-like screening pipeline.
    """

    matches = training_df[
        training_df["kepoi_name"].astype(str).str.upper()
        == kepoi_name.upper()
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate {kepoi_name} was not found in the training dataset."
        )

    row = matches.iloc[0]

    # --------------------------------------------------------
    # Build model feature dictionary
    # --------------------------------------------------------

    model_features = {}

    for feature in FEATURES:
        if feature not in row.index:
            raise HTTPException(
                status_code=500,
                detail=f"Required model feature '{feature}' is missing."
            )

        value = row[feature]

        if pd.isna(value):
            model_features[feature] = None
        else:
            model_features[feature] = float(value)

    # --------------------------------------------------------
    # Validate model features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature, value in model_features.items()
        if value is None
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Candidate contains missing model features.",
                "missing_features": missing_features,
            },
        )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    X = pd.DataFrame(
        [[model_features[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(model.predict(X)[0])

    # --------------------------------------------------------
    # Screening values
    # --------------------------------------------------------

    try:
        prad = float(row["koi_prad"])
        insol = float(row["koi_insol"])
        teq = float(row["koi_teq"])
        period = float(row["koi_period"])
        steff = float(row["koi_steff"])

        rho_ratio = float(row["f_rho_ratio"])
        duration_ratio = float(row["f_duration_ratio"])
        duration_unphysical = bool(row["f_duration_unphysical"])
        depth_ratio = float(row["f_depth_ratio"])

        log_rho_ratio = float(row["f_log_rho_ratio"])
        impact_discrepancy = float(row["f_impact_discrepancy"])

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read screening features: {exc}"
        )

    # --------------------------------------------------------
    # Basic Earth-like screening
    # --------------------------------------------------------

    passes_radius = 0.8 <= prad <= 1.25
    passes_insolation = 0.5 <= insol <= 2.0
    passes_probability = probability >= 0.80

    # --------------------------------------------------------
    # Physics consistency
    # --------------------------------------------------------

    density_consistent = 0.7 <= rho_ratio <= 1.3

    duration_consistent = (
        0.5 <= duration_ratio <= 1.5
        and not duration_unphysical
    )

    depth_consistent = 0.7 <= depth_ratio <= 1.3

    overall_physics_consistent = (
        density_consistent
        and duration_consistent
        and depth_consistent
    )

    # --------------------------------------------------------
    # Transit consistency score
    # --------------------------------------------------------

    transit_consistency_score = (
        np.exp(-abs(log_rho_ratio))
        * np.exp(-abs(impact_discrepancy))
        * np.exp(-abs(np.log(max(depth_ratio, 1e-12))))
    )

    transit_consistency_score = float(
        np.clip(transit_consistency_score, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # Project-defined Earth-like score
    # --------------------------------------------------------

    radius_score = np.exp(
        -0.5 * ((prad - 1.0) / 0.30) ** 2
    )

    insolation_score = np.exp(
        -0.5 * (
            np.log10(max(insol, 1e-12))
            / np.log10(2)
        ) ** 2
    )

    temperature_score = np.exp(
        -0.5 * ((teq - 255) / 80) ** 2
    )

    stellar_temperature_score = np.exp(
        -0.5 * ((steff - 5772) / 1000) ** 2
    )

    period_score = np.exp(
        -0.5 * (
            np.log10(max(period, 1e-12) / 365.25)
            / 0.8
        ) ** 2
    )

    earthlike_score = (
        0.30 * probability
        + 0.20 * radius_score
        + 0.20 * insolation_score
        + 0.10 * temperature_score
        + 0.05 * stellar_temperature_score
        + 0.05 * period_score
        + 0.10 * transit_consistency_score
    )

    earthlike_score = float(
        np.clip(earthlike_score, 0.0, 1.0)
    )

    passes_screen = (
        passes_radius
        and passes_insolation
        and passes_probability
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "kepoi_name": str(row["kepoi_name"]),
        "koi_pdisposition": str(row["koi_pdisposition"])
        if "koi_pdisposition" in row.index
        else None,

        "prediction": prediction,
        "planet_probability": probability,

        "passes_screen": passes_screen,

        "screening_criteria": {
            "radius": {
                "value": prad,
                "minimum": 0.8,
                "maximum": 1.25,
                "passes": passes_radius,
            },
            "insolation": {
                "value": insol,
                "minimum": 0.5,
                "maximum": 2.0,
                "passes": passes_insolation,
            },
            "planet_probability": {
                "value": probability,
                "minimum": 0.80,
                "passes": passes_probability,
            },
        },

        "physics_consistency": {
            "density_consistent": density_consistent,
            "duration_consistent": duration_consistent,
            "depth_consistent": depth_consistent,
            "overall_physics_consistent": overall_physics_consistent,
        },

        "scores": {
            "earthlike_screening_score": earthlike_score,
            "transit_consistency_score": transit_consistency_score,
            "radius_score": float(radius_score),
            "insolation_score": float(insolation_score),
            "temperature_score": float(temperature_score),
            "stellar_temperature_score": float(
                stellar_temperature_score
            ),
            "period_score": float(period_score),
        },

        "physical_values": {
            "radius_earth_radii": prad,
            "insolation_earth_flux": insol,
            "equilibrium_temperature_k": teq,
            "orbital_period_days": period,
            "stellar_temperature_k": steff,
            "density_ratio": rho_ratio,
            "duration_ratio": duration_ratio,
            "depth_ratio": depth_ratio,
        },
    }

# ============================================================
# EARTH-LIKE SCREENING
# ============================================================

@app.post("/api/screen")
def screen_planet(request: PredictionRequest):

    supplied = request.features

    required_screening_features = [
        "koi_prad",
        "koi_insol",
        "koi_teq",
        "koi_period",
        "koi_steff",
        "f_rho_ratio",
        "f_duration_ratio",
        "f_duration_unphysical",
        "f_depth_ratio",
        "f_log_rho_ratio",
        "f_impact_discrepancy",
    ]

    missing = [
        feature
        for feature in required_screening_features
        if feature not in supplied
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing screening features.",
                "missing_features": missing,
            },
        )

    # ----------------------------
    # Model probability
    # ----------------------------

    values = {}

    for feature in FEATURES:
        value = supplied.get(feature)

        if value is None:
            values[feature] = np.nan
        else:
            try:
                values[feature] = float(value)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Feature '{feature}' must be numeric "
                        "or null."
                    ),
                )

    input_df = pd.DataFrame(
        [values],
        columns=FEATURES,
    )

    try:
        probability = float(
            model.predict_proba(input_df)[0][1]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )

    # ----------------------------
    # Earth-like screen
    # ----------------------------

    prad = float(supplied["koi_prad"])
    insol = float(supplied["koi_insol"])

    passes_radius = 0.8 <= prad <= 1.25
    passes_insolation = 0.5 <= insol <= 2.0
    passes_probability = probability >= 0.80

    passes_screen = (
        passes_radius
        and passes_insolation
        and passes_probability
    )

    # ----------------------------
    # Physics consistency
    # ----------------------------

    rho_ratio = float(supplied["f_rho_ratio"])

    duration_ratio = float(
        supplied["f_duration_ratio"]
    )

    duration_unphysical = bool(
        supplied["f_duration_unphysical"]
    )

    depth_ratio = float(
        supplied["f_depth_ratio"]
    )

    density_consistent = (
        0.7 <= rho_ratio <= 1.3
    )

    duration_consistent = (
        0.5 <= duration_ratio <= 1.5
        and not duration_unphysical
    )

    depth_consistent = (
        0.7 <= depth_ratio <= 1.3
    )

    overall_physics_consistent = (
        density_consistent
        and duration_consistent
        and depth_consistent
    )

    # ----------------------------
    # Transit consistency
    # ----------------------------

    log_rho_ratio = float(
        supplied["f_log_rho_ratio"]
    )

    impact_discrepancy = float(
        supplied["f_impact_discrepancy"]
    )

    transit_score = (
        np.exp(-abs(log_rho_ratio))
        * np.exp(-abs(impact_discrepancy))
        * np.exp(-abs(np.log(depth_ratio)))
    )

    transit_score = float(
        np.clip(transit_score, 0, 1)
    )

    # ----------------------------
    # Project screening score
    # ----------------------------

    teq = float(supplied["koi_teq"])
    period = float(supplied["koi_period"])
    steff = float(supplied["koi_steff"])

    radius_score = float(
        np.exp(
            -0.5
            * ((prad - 1.0) / 0.30) ** 2
        )
    )

    insolation_score = float(
        np.exp(
            -0.5
            * (
                np.log10(insol)
                / np.log10(2)
            ) ** 2
        )
    )

    temperature_score = float(
        np.exp(
            -0.5
            * ((teq - 255) / 80) ** 2
        )
    )

    stellar_temperature_score = float(
        np.exp(
            -0.5
            * ((steff - 5772) / 1000) ** 2
        )
    )

    period_score = float(
        np.exp(
            -0.5
            * (
                np.log10(period / 365.25)
                / 0.8
            ) ** 2
        )
    )

    earthlike_score = float(
        0.30 * probability
        + 0.20 * radius_score
        + 0.20 * insolation_score
        + 0.10 * temperature_score
        + 0.05 * stellar_temperature_score
        + 0.05 * period_score
        + 0.10 * transit_score
    )

    return {
        "planet_probability": probability,
        "passes_screen": passes_screen,
        "screening_criteria": {
            "radius": {
                "value": prad,
                "minimum": 0.8,
                "maximum": 1.25,
                "passes": passes_radius,
            },
            "insolation": {
                "value": insol,
                "minimum": 0.5,
                "maximum": 2.0,
                "passes": passes_insolation,
            },
            "planet_probability": {
                "value": probability,
                "minimum": 0.80,
                "passes": passes_probability,
            },
        },
        "physics_consistency": {
            "density_consistent": density_consistent,
            "duration_consistent": duration_consistent,
            "depth_consistent": depth_consistent,
            "overall_physics_consistent": (
                overall_physics_consistent
            ),
        },
        "scores": {
            "earthlike_screening_score": earthlike_score,
            "transit_consistency_score": transit_score,
            "radius_score": radius_score,
            "insolation_score": insolation_score,
            "temperature_score": temperature_score,
            "stellar_temperature_score": (
                stellar_temperature_score
            ),
            "period_score": period_score,
        },
        
    }