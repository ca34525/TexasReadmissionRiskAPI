# tests/test_main.py

from fastapi.testclient import TestClient
import pytest

# Import the FastAPI app instance from your main.py file
from main import app

# Create a TestClient instance
client = TestClient(app)

# --- Tests for the new Interactive (`POST`) Endpoint ---

def test_post_interactive_prediction_success():
    """
    Test the happy path for the interactive prediction endpoint.
    Sends a valid payload and expects a successful 200 response.
    """
    # This is a sample payload that matches the PredictionFeatures Pydantic model
    test_payload = {
        "length_of_stay": 5, "age_at_admission": 65, "gender": "F",
        "race": "white", "marital_status": "M", "admission_reason": "Pneumonia",
        "payer": "Medicare", "total_claim_cost": 12000.0, "income": 75000,
        "admission_day_of_week": "Friday", "primary_diagnosis_code": "J18.9",
        "provider_id": "prov-xyz", "prior_admissions_last_year": 0,
        "num_diagnoses": 2, "num_procedures": 0, "num_medications": 1
    }
    
    response = client.post("/predict/interactive", json=test_payload)
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "readmission_probability" in data
    assert "prediction" in data
    assert "threshold" in data
    assert 0.0 <= data["readmission_probability"] <= 1.0
    assert data["prediction"] in [0, 1]

def test_post_interactive_prediction_validation_error():
    """
    Test the validation error path. Sends a payload with a missing required field.
    FastAPI should automatically catch this and return a 422 Unprocessable Entity error.
    """
    # This payload is missing the required 'length_of_stay' field
    invalid_payload = {
        "age_at_admission": 65, "gender": "F",
        "race": "white", "marital_status": "M", "admission_reason": "Pneumonia",
        "payer": "Medicare", "total_claim_cost": 12000.0, "income": 75000,
        "admission_day_of_week": "Friday", "primary_diagnosis_code": "J18.9",
        "provider_id": "prov-xyz", "prior_admissions_last_year": 0,
        "num_diagnoses": 2, "num_procedures": 0, "num_medications": 1
    }

    response = client.post("/predict/interactive", json=invalid_payload)
    
    # Assert that FastAPI's validation is working
    assert response.status_code == 422


# --- Tests for the ID-Based (`GET`) Endpoint ---

def test_get_prediction_not_found(monkeypatch):
    """
    Tests the failure case for the GET endpoint where an encounter_id is not found.
    We use monkeypatch to simulate the behavior of `make_prediction` without
    actually hitting the database.
    """
    # We are telling pytest: "When `main.make_prediction` is called,
    # don't run the real function. Instead, just return this dictionary."
    def mock_make_prediction_error(encounter_id):
        return {"error": "Encounter ID not found."}

    monkeypatch.setattr("main.make_prediction", mock_make_prediction_error)
    
    response = client.get("/predict/some-fake-id")
    
    assert response.status_code == 404
    assert response.json() == {"detail": "Encounter ID not found."}


# NOTE: A success test for the GET endpoint would require a live test database,
# which is more complex to set up. Mocking the failure case is a good,
# lightweight way to ensure the API's error handling works correctly.
