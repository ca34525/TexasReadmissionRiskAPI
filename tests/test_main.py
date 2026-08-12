import pytest
from fastapi.testclient import TestClient

import main
from src.predict import (
    DATABASE_UNAVAILABLE,
    ENCOUNTER_NOT_FOUND,
    MODEL_UNAVAILABLE,
    PREDICTION_FAILED,
)

client = TestClient(main.app)


@pytest.fixture
def valid_payload():
    return {
        "length_of_stay": 5,
        "age_at_admission": 65,
        "gender": "F",
        "race": "white",
        "marital_status": "M",
        "admission_reason": "Pneumonia",
        "payer": "Medicare",
        "total_claim_cost": 12000.0,
        "income": 75000,
        "admission_day_of_week": "Friday",
        "primary_diagnosis_code": "J18.9",
        "provider_id": "prov-xyz",
        "prior_admissions_last_year": 0,
        "num_diagnoses": 2,
        "num_procedures": 0,
        "num_medications": 1,
    }


class StubModel:
    feature_names_ = [
        "length_of_stay",
        "age_at_admission",
        "gender",
        "race",
        "marital_status",
        "admission_reason",
        "payer",
        "total_claim_cost",
        "income",
        "admission_day_of_week",
        "primary_diagnosis_code",
        "provider_id",
        "payer_dx_interaction",
        "prior_admissions_last_year",
        "num_diagnoses",
        "num_procedures",
        "num_medications",
    ]

    def predict_proba(self, features):
        assert list(features.columns) == self.feature_names_
        return {(0, 1): 0.8}


def test_post_interactive_prediction_success(monkeypatch, valid_payload):
    monkeypatch.setattr(main, "model", StubModel())
    monkeypatch.setattr(main, "THRESHOLD", 0.7)

    response = client.post("/predict/interactive", json=valid_payload)

    assert response.status_code == 200
    assert response.json() == {
        "readmission_probability": 0.8,
        "prediction": 1,
        "threshold": 0.7,
    }


def test_post_interactive_prediction_validation_error(valid_payload):
    valid_payload.pop("length_of_stay")

    response = client.post("/predict/interactive", json=valid_payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age_at_admission", -1),
        ("length_of_stay", -1),
        ("total_claim_cost", -0.01),
        ("num_medications", -1),
    ],
)
def test_post_interactive_prediction_rejects_negative_values(
    valid_payload, field, value
):
    valid_payload[field] = value

    response = client.post("/predict/interactive", json=valid_payload)

    assert response.status_code == 422


def test_post_interactive_prediction_rejects_extra_fields(valid_payload):
    valid_payload["unsupported_feature"] = 1

    response = client.post("/predict/interactive", json=valid_payload)

    assert response.status_code == 422


def test_post_interactive_prediction_returns_503_without_model(
    monkeypatch, valid_payload
):
    monkeypatch.setattr(main, "model", None)

    response = client.post("/predict/interactive", json=valid_payload)

    assert response.status_code == 503
    assert response.json() == {"detail": "Model artifacts are unavailable."}


def test_liveness_is_independent_of_dependencies(monkeypatch):
    monkeypatch.setattr(main, "model", None)

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_success(monkeypatch):
    monkeypatch.setattr(main, "model", StubModel())
    monkeypatch.setattr(main, "database_is_available", lambda: True)

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_returns_503_without_model(monkeypatch):
    monkeypatch.setattr(main, "model", None)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Model artifacts are unavailable."}


def test_readiness_returns_503_without_database(monkeypatch):
    monkeypatch.setattr(main, "model", StubModel())
    monkeypatch.setattr(main, "database_is_available", lambda: False)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction database is unavailable."}


def test_readiness_returns_503_when_database_check_fails(monkeypatch):
    def failed_database_check():
        raise RuntimeError("internal details")

    monkeypatch.setattr(main, "model", StubModel())
    monkeypatch.setattr(main, "database_is_available", failed_database_check)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction database is unavailable."}


@pytest.mark.parametrize(
    ("error_code", "expected_status"),
    [
        (ENCOUNTER_NOT_FOUND, 404),
        (MODEL_UNAVAILABLE, 503),
        (DATABASE_UNAVAILABLE, 503),
        (PREDICTION_FAILED, 500),
        ("unknown_error", 500),
    ],
)
def test_get_prediction_maps_internal_errors(monkeypatch, error_code, expected_status):
    monkeypatch.setattr(
        main,
        "make_prediction",
        lambda encounter_id: {
            "error": "Prediction error.",
            "error_code": error_code,
        },
    )

    response = client.get("/predict/some-fake-id")

    assert response.status_code == expected_status
    assert response.json() == {"detail": "Prediction error."}


def test_get_prediction_not_found_preserves_message(monkeypatch):
    monkeypatch.setattr(
        main,
        "make_prediction",
        lambda encounter_id: {
            "error": "Encounter ID not found.",
            "error_code": ENCOUNTER_NOT_FOUND,
        },
    )

    response = client.get("/predict/some-fake-id")

    assert response.status_code == 404
    assert response.json() == {"detail": "Encounter ID not found."}


def test_get_prediction_handles_unexpected_exception(monkeypatch):
    def raise_unexpected_error(encounter_id):
        raise RuntimeError("internal details")

    monkeypatch.setattr(main, "make_prediction", raise_unexpected_error)

    response = client.get("/predict/some-fake-id")

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction could not be generated."}
