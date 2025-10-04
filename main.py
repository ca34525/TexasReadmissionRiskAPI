# main.py

import json
import logging
from pathlib import Path

import catboost as cb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Import project modules ---
# We need the config for file paths and feature lists
from src import config
# We still need the original prediction logic for the ID-based endpoint
from src.predict import make_prediction

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Load Model and Metadata at Startup ---
# This is a performance optimization.  By loading the model and metadata once
# when the application starts, we avoid reloading them for every single API request.
try:
    model = cb.CatBoostClassifier()
    model.load_model(str(config.MODEL_FILE))

    metadata_path = config.MODEL_FILE.parent / "model_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    THRESHOLD = metadata["optimal_threshold"]
    logging.info("Model and metadata loaded successfully at startup.")
except FileNotFoundError as e:
    logging.error(f"FATAL: Model or metadata not found at startup: {e}")
    # In a real application, you might exit here or handle it more gracefully
    model = None
    THRESHOLD = 0.7 # A default fallback

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
    length_of_stay: int = Field(..., json_schema_extra={'example': 7})
    age_at_admission: int = Field(..., json_schema_extra={'example': 50})
    gender: str = Field(..., json_schema_extra={'example': "male"})
    race: str = Field(..., json_schema_extra={'example': "White"})
    marital_status: str = Field(..., json_schema_extra={'example': "M"})
    admission_reason: str = Field(..., json_schema_extra={'example': "Encounter for problem (procedure)"})
    payer: str = Field(..., json_schema_extra={'example': "Medicare"})
    total_claim_cost: float = Field(..., json_schema_extra={'example': 26483})
    income: int = Field(..., json_schema_extra={'example': 74739})
    admission_day_of_week: str = Field(..., json_schema_extra={'example': "Tuesday"})
    primary_diagnosis_code: str = Field(..., json_schema_extra={'example': "424132000"})
    provider_id: str = Field(..., json_schema_extra={'example': "us-npi|9999868992"})
    prior_admissions_last_year: int = Field(..., json_schema_extra={'example': 2})
    num_diagnoses: int = Field(..., json_schema_extra={'example': 1})
    num_procedures: int = Field(..., json_schema_extra={'example': 9})
    num_medications: int = Field(..., json_schema_extra={'example': 1})

class PredictionResponse(BaseModel):
    """Defines the structure for the ID-based prediction response."""
    encounter_id: str
    readmission_probability: float
    prediction: int
    threshold: float

class InteractivePredictionResponse(BaseModel):
    """Defines the structure for the interactive prediction response."""
    readmission_probability: float = Field(..., json_schema_extra={'example': 0.8245})
    prediction: int = Field(..., json_schema_extra={'example': 1}, description="1 for high risk, 0 for low risk.")
    threshold: float = Field(..., json_schema_extra={'example': 0.7})

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome! Navigate to /docs for API documentation."}


@app.get("/predict/{encounter_id}", response_model=PredictionResponse, tags=["ID-Based Prediction"])
def get_prediction(encounter_id: str):
    """
    Predicts readmission risk based on a historical encounter_id.
    """
    logging.info(f"Received ID-based prediction request for: {encounter_id}")
    result = make_prediction(encounter_id)
    if "error" in result:
        logging.error(f"Prediction failed for {encounter_id}: {result['error']}")
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/predict/interactive", response_model=InteractivePredictionResponse, tags=["Interactive Prediction"])
def post_interactive_prediction(features: PredictionFeatures):
    """
    Predicts readmission risk from a JSON payload of patient features.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    logging.info("Received interactive prediction request.")
    
    # 1.  Convert the Pydantic model to a dictionary
    features_dict = features.model_dump()
    
    # 2.  Engineer the interaction feature, just like in training
    features_dict["payer_dx_interaction"] = (
        str(features_dict.get("payer", "unknown")) + "_" +
        str(features_dict.get("primary_diagnosis_code", "unknown"))
    )
    
    # 3.  Create a single-row DataFrame
    df = pd.DataFrame([features_dict])
    
    # 4.  Preprocess categorical features exactly as done in training
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("missing").astype("category")

    # 5.  Ensure column order matches the model's expectation
    df = df.reindex(columns=model.feature_names_, fill_value=0)
    
    # 6.  Make the prediction
    pred_proba = model.predict_proba(df)[0, 1]
    prediction = 1 if pred_proba >= THRESHOLD else 0

    logging.info(f"Generated interactive prediction.  Probability: {pred_proba:.4f}")
    
    return {
        "readmission_probability": pred_proba,
        "prediction": prediction,
        "threshold": THRESHOLD,
    }