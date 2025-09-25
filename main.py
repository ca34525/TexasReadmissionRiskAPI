# main.py

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Core Prediction Logic ---
# We import the function that contains all the logic for making a prediction.
# The API's job is simply to expose this function to the web.
from src.predict import make_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# --- API Initialization ---
# Create a FastAPI application instance. This is the main point of interaction.
app = FastAPI(
    title="Readmission Prediction API",
    description="An API to predict the risk of hospital readmission for a given patient encounter.",
    version="1.0.0",
)


# --- Pydantic Models (Data Validation) ---
# Pydantic models define the structure and data types for your API's inputs
# and outputs. FastAPI uses them to validate incoming requests and format
# outgoing responses, which helps prevent errors and provides great documentation.

class PredictionResponse(BaseModel):
    """Defines the structure of the JSON response for a prediction."""
    encounter_id: str = Field(..., example="a933a39e-b98f-4171-8b9a-8a0a861d3e1d")
    readmission_probability: float = Field(..., example=0.8245)
    prediction: int = Field(..., example=1, description="1 for high risk, 0 for low risk.")
    threshold: float = Field(..., example=0.7)


# --- API Endpoints ---
# Endpoints are the specific URLs that users of your API will access.
# The decorator (@app.get, @app.post, etc.) tells FastAPI how to handle
# requests to that URL.

@app.get("/", tags=["General"])
def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "Welcome to the Readmission Prediction API. Go to /docs for more info."}


@app.get("/predict/{encounter_id}", response_model=PredictionResponse, tags=["Prediction"])
def get_prediction(encounter_id: str):
    """
    Accepts an encounter_id, fetches data, engineers features, and
    returns a real-time readmission risk score.
    """
    logging.info(f"Received prediction request for encounter_id: {encounter_id}")

    # Call our core logic from the predict.py script
    result = make_prediction(encounter_id)

    # Handle cases where the prediction logic returns an error
    # (e.g., encounter_id not found). We turn this into a standard
    # HTTP error response.
    if "error" in result:
        logging.error(f"Prediction failed for {encounter_id}: {result['error']}")
        raise HTTPException(
            status_code=404,
            detail=result["error"]
        )

    logging.info(f"Successfully generated prediction for {encounter_id}")
    return result
