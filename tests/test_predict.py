# tests/test_predict.py

import json

import catboost as cb
import duckdb
import pandas as pd
import pytest

# Import the function to be tested and the config it uses
from src import config
from src.predict import (
    DATABASE_UNAVAILABLE,
    ENCOUNTER_NOT_FOUND,
    MODEL_UNAVAILABLE,
    database_is_available,
    list_inpatient_encounter_ids,
    make_prediction,
)


@pytest.fixture
def setup_test_environment(tmp_path, monkeypatch):
    """
    Creates a temporary, self-contained environment for testing.
    This includes a dummy database, a dummy model, and mocks the config.
    """
    # 1. Create temporary directories
    temp_output_dir = tmp_path / "output"
    temp_models_dir = tmp_path / "models"
    temp_output_dir.mkdir()
    temp_models_dir.mkdir()

    # 2. Define temporary file paths
    db_file = temp_output_dir / "test.duckdb"
    model_file = temp_models_dir / "test_model.cbm"
    metadata_file = temp_models_dir / "model_metadata.json"

    # 3. Use monkeypatch to make the config use our temporary paths
    monkeypatch.setattr(config, "DB_FILE", db_file)
    monkeypatch.setattr(config, "MODEL_FILE", model_file)
    monkeypatch.setattr(config, "MODELS_DIR", temp_models_dir)

    # 4. Create and populate a dummy DuckDB database
    con = duckdb.connect(database=str(db_file), read_only=False)
    encounter_id_to_test = "enc-inp-001"
    patient_id_for_encounter = "pat-001"

    con.execute(
        "CREATE TABLE patients (Id VARCHAR, BirthDate DATE, Gender VARCHAR, Race VARCHAR, Marital VARCHAR, Income BIGINT);"
    )
    con.execute(
        "INSERT INTO patients VALUES (?, '1970-01-01', 'F', 'white', 'M', 75000);",
        [patient_id_for_encounter],
    )

    con.execute("""
        CREATE TABLE encounters (
            Id VARCHAR, Patient VARCHAR, Start TIMESTAMP, Stop TIMESTAMP,
            EncounterClass VARCHAR, Description VARCHAR, Payer VARCHAR,
            Total_Claim_Cost FLOAT, ReasonCode VARCHAR, Provider VARCHAR
        );
    """)
    con.execute(
        "INSERT INTO encounters VALUES (?, ?, '2025-01-10', '2025-01-15', 'IMP', 'Pneumonia', 'Medicare', 12000.0, 'J18.9', 'prov-xyz');",
        [encounter_id_to_test, patient_id_for_encounter],
    )

    con.execute("CREATE TABLE conditions (Encounter VARCHAR, Code VARCHAR);")
    con.execute(
        "INSERT INTO conditions VALUES (?, 'J18.9'), (?, 'E11.9');",
        [encounter_id_to_test, encounter_id_to_test],
    )
    con.execute("CREATE TABLE procedures (Encounter VARCHAR, Code VARCHAR);")
    con.execute("CREATE TABLE medications (Encounter VARCHAR, Code VARCHAR);")
    con.execute("INSERT INTO medications VALUES (?, 'med123');", [encounter_id_to_test])

    con.close()

    # 5. Create and save a dummy CatBoost model
    dummy_features = {
        "length_of_stay": [5, 10],
        "age_at_admission": [55, 65],
        "gender": ["F", "M"],
        "race": ["white", "black"],
        "marital_status": ["M", "S"],
        "admission_reason": ["Pneumonia", "Heart Failure"],
        "payer": ["Medicare", "Aetna"],
        "total_claim_cost": [12000.0, 25000.0],
        "income": [75000, 85000],
        "admission_day_of_week": ["Friday", "Monday"],
        "primary_diagnosis_code": ["J18.9", "I50.9"],
        "provider_id": ["prov-xyz", "prov-abc"],
        "payer_dx_interaction": ["Medicare_J18.9", "Aetna_I50.9"],
        "prior_admissions_last_year": [0, 1],
        "num_diagnoses": [2, 5],
        "num_procedures": [0, 1],
        "num_medications": [1, 4],
    }
    X_train = pd.DataFrame(dummy_features)
    y_train = pd.Series([1, 0])

    # Ensure categorical features are treated as strings
    for col in config.CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)

    dummy_model = cb.CatBoostClassifier(
        iterations=1,
        random_seed=42,
        thread_count=1,
        verbose=0,
        allow_writing_files=False,
    )
    dummy_model.fit(X_train, y_train, cat_features=config.CATEGORICAL_FEATURES)
    dummy_model.save_model(str(model_file))

    # 6. Create dummy model metadata
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump({"optimal_threshold": 0.5}, f)

    yield encounter_id_to_test


def test_make_prediction_success(setup_test_environment):
    """
    Tests the happy path: a valid encounter ID that exists in the DB.
    """
    encounter_id = setup_test_environment
    result = make_prediction(encounter_id)

    assert isinstance(result, dict)
    assert "error" not in result
    assert result["encounter_id"] == encounter_id
    assert "readmission_probability" in result
    assert "prediction" in result
    assert "threshold" in result
    assert isinstance(result["readmission_probability"], float)
    assert 0.0 <= result["readmission_probability"] <= 1.0
    assert result["prediction"] in [0, 1]
    assert result["threshold"] == 0.5


def test_database_readiness_and_encounter_choices(setup_test_environment):
    assert database_is_available() is True
    assert list_inpatient_encounter_ids() == [setup_test_environment]


def test_database_readiness_rejects_missing_required_column(
    setup_test_environment, tmp_path, monkeypatch
):
    invalid_db = tmp_path / "invalid.duckdb"
    con = duckdb.connect(str(invalid_db))
    con.execute("CREATE TABLE encounters (Id VARCHAR)")
    con.close()
    monkeypatch.setattr(config, "DB_FILE", invalid_db)

    assert database_is_available() is False


def test_make_prediction_not_found(setup_test_environment):
    """
    Tests the failure path: an encounter ID that does not exist.
    """
    non_existent_id = "enc-id-that-does-not-exist"
    result = make_prediction(non_existent_id)

    assert isinstance(result, dict)
    assert "error" in result
    assert (
        result["error"]
        == f"Could not generate features for encounter ID {non_existent_id}."
    )
    assert result["error_code"] == ENCOUNTER_NOT_FOUND


def test_make_prediction_handles_database_connection_failure(
    setup_test_environment, monkeypatch
):
    def unavailable_database(*args, **kwargs):
        raise duckdb.IOException("database unavailable")

    monkeypatch.setattr("src.predict.duckdb.connect", unavailable_database)

    result = make_prediction(setup_test_environment)

    assert result == {
        "error": "Prediction database is unavailable.",
        "error_code": DATABASE_UNAVAILABLE,
    }


def test_make_prediction_handles_missing_model(setup_test_environment):
    config.MODEL_FILE.unlink()

    result = make_prediction(setup_test_environment)

    assert result == {
        "error": "Model artifacts are unavailable.",
        "error_code": MODEL_UNAVAILABLE,
    }


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, "0.5", None])
def test_make_prediction_rejects_invalid_threshold(setup_test_environment, threshold):
    metadata_path = config.MODEL_FILE.parent / "model_metadata.json"
    metadata_path.write_text(
        json.dumps({"optimal_threshold": threshold}), encoding="utf-8"
    )

    result = make_prediction(setup_test_environment)

    assert result == {
        "error": "Model artifacts are unavailable.",
        "error_code": MODEL_UNAVAILABLE,
    }


def test_make_prediction_handles_corrupt_metadata(setup_test_environment):
    metadata_path = config.MODEL_FILE.parent / "model_metadata.json"
    metadata_path.write_text("{not-json", encoding="utf-8")

    result = make_prediction(setup_test_environment)

    assert result == {
        "error": "Model artifacts are unavailable.",
        "error_code": MODEL_UNAVAILABLE,
    }
