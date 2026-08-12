# main.py

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# --- Import project modules ---
from src import config
from src.inference import predict_feature_mapping
from src.model_artifacts import (
    ModelArtifactsUnavailableError,
    load_model_artifacts,
)
from src.predict import (
    DATABASE_UNAVAILABLE,
    ENCOUNTER_NOT_FOUND,
    ERROR_CODE_KEY,
    MODEL_UNAVAILABLE,
    PREDICTION_FAILED,
    database_is_available,
    make_prediction,
)

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Load Model and Metadata at Startup ---
# This is a performance optimization.  By loading the model and metadata once
# when the application starts, we avoid reloading them for every single API request.
try:
    artifacts = load_model_artifacts(config.MODEL_FILE)
except ModelArtifactsUnavailableError:
    logger.exception("Model and metadata could not be loaded at startup.")
    model = None
    THRESHOLD = float(config.FINAL_THRESHOLD)
else:
    model = artifacts.model
    THRESHOLD = artifacts.threshold
    logger.info("Model and metadata loaded successfully at startup.")

# --- API Initialization ---
app = FastAPI(
    title="Readmission Prediction API",
    description="An API to predict hospital readmission risk.",
    version="1.0.0",
)


# --- Pydantic Models (Data Validation) ---
class PredictionFeatures(BaseModel):
    """
    Defines the structure for the JSON payload for the interactive endpoint.
    These are all the features the model needs to make a prediction.
    FastAPI will automatically validate that incoming data matches this structure.
    """

    model_config = ConfigDict(extra="forbid")

    length_of_stay: int = Field(ge=0, json_schema_extra={"example": 7})
    age_at_admission: int = Field(ge=0, le=130, json_schema_extra={"example": 50})
    gender: str = Field(
        min_length=1, max_length=50, json_schema_extra={"example": "male"}
    )
    race: str = Field(
        min_length=1, max_length=100, json_schema_extra={"example": "White"}
    )
    marital_status: str = Field(
        min_length=1, max_length=50, json_schema_extra={"example": "M"}
    )
    admission_reason: str = Field(
        min_length=1,
        max_length=500,
        json_schema_extra={"example": "Encounter for problem (procedure)"},
    )
    payer: str = Field(
        min_length=1, max_length=200, json_schema_extra={"example": "Medicare"}
    )
    total_claim_cost: float = Field(
        ge=0, allow_inf_nan=False, json_schema_extra={"example": 26483}
    )
    income: int = Field(ge=0, json_schema_extra={"example": 74739})
    admission_day_of_week: str = Field(
        min_length=1, max_length=20, json_schema_extra={"example": "Tuesday"}
    )
    primary_diagnosis_code: str = Field(
        min_length=1, max_length=100, json_schema_extra={"example": "424132000"}
    )
    provider_id: str = Field(
        min_length=1,
        max_length=256,
        json_schema_extra={"example": "us-npi|9999868992"},
    )
    prior_admissions_last_year: int = Field(ge=0, json_schema_extra={"example": 2})
    num_diagnoses: int = Field(ge=0, json_schema_extra={"example": 1})
    num_procedures: int = Field(ge=0, json_schema_extra={"example": 9})
    num_medications: int = Field(ge=0, json_schema_extra={"example": 1})


class PredictionResponse(BaseModel):
    """Defines the structure for the ID-based prediction response."""

    encounter_id: str
    readmission_probability: float
    prediction: int
    threshold: float


class InteractivePredictionResponse(BaseModel):
    """Defines the structure for the interactive prediction response."""

    readmission_probability: float = Field(..., json_schema_extra={"example": 0.8245})
    prediction: int = Field(
        ...,
        json_schema_extra={"example": 1},
        description="1 for high risk, 0 for low risk.",
    )
    threshold: float = Field(..., json_schema_extra={"example": 0.7})


# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome! Navigate to /docs for API documentation."}


@app.get("/health/live", tags=["Health"])
def get_liveness():
    """Report whether the API process is running."""
    return {"status": "ok"}


@app.get("/health/ready", tags=["Health"])
def get_readiness():
    """Report whether model artifacts and the serving database are usable."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifacts are unavailable.")
    try:
        database_ready = database_is_available()
    except Exception:
        logger.exception("Unexpected database readiness-check failure.")
        database_ready = False
    if not database_ready:
        raise HTTPException(
            status_code=503, detail="Prediction database is unavailable."
        )
    return {"status": "ready"}


@app.get(
    "/predict/{encounter_id}",
    response_model=PredictionResponse,
    tags=["ID-Based Prediction"],
)
def get_prediction(encounter_id: str):
    """
    Predicts readmission risk based on a historical encounter_id.
    """
    logger.info("Received ID-based prediction request for: %s", encounter_id)
    try:
        result = make_prediction(encounter_id)
    except Exception as exc:
        logger.exception("Unexpected prediction failure for %s", encounter_id)
        raise HTTPException(
            status_code=500, detail="Prediction could not be generated."
        ) from exc

    if "error" in result:
        error_code = result.get(ERROR_CODE_KEY, PREDICTION_FAILED)
        status_code = {
            ENCOUNTER_NOT_FOUND: 404,
            MODEL_UNAVAILABLE: 503,
            DATABASE_UNAVAILABLE: 503,
            PREDICTION_FAILED: 500,
        }.get(error_code, 500)
        logger.error(
            "Prediction failed for %s (%s): %s",
            encounter_id,
            error_code,
            result["error"],
        )
        raise HTTPException(status_code=status_code, detail=result["error"])
    return result


@app.post(
    "/predict/interactive",
    response_model=InteractivePredictionResponse,
    tags=["Interactive Prediction"],
)
def post_interactive_prediction(features: PredictionFeatures):
    """
    Predicts readmission risk from a JSON payload of patient features.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifacts are unavailable.")

    logger.info("Received interactive prediction request.")

    try:
        result = predict_feature_mapping(model, THRESHOLD, features.model_dump())
    except Exception as exc:
        logger.exception("Interactive prediction failed.")
        raise HTTPException(
            status_code=500, detail="Prediction could not be generated."
        ) from exc

    logger.info(
        "Generated interactive prediction. Probability: %.4f",
        result["readmission_probability"],
    )
    return result
