# src/predict.py

import argparse
import json
import logging

import duckdb
import pandas as pd

# Import variables from our central configuration
from . import config
from .inference import predict_feature_frame
from .model_artifacts import (
    ModelArtifactsUnavailableError,
    load_model_artifacts,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


ERROR_CODE_KEY = "error_code"
ENCOUNTER_NOT_FOUND = "encounter_not_found"
MODEL_UNAVAILABLE = "model_unavailable"
DATABASE_UNAVAILABLE = "database_unavailable"
PREDICTION_FAILED = "prediction_failed"

_REQUIRED_PREDICTION_COLUMNS = {
    "encounters": (
        "Id",
        "Patient",
        "Start",
        "Stop",
        "EncounterClass",
        "Description",
        "Payer",
        "Total_Claim_Cost",
        "ReasonCode",
        "Provider",
    ),
    "patients": ("Id", "BirthDate", "Gender", "Race", "Marital", "Income"),
    "conditions": ("Encounter", "Code"),
    "procedures": ("Encounter", "Code"),
    "medications": ("Encounter", "Code"),
}


def _error_result(message: str, code: str) -> dict:
    """Return an error shape compatible with the existing Gradio consumer."""
    return {"error": message, ERROR_CODE_KEY: code}


def database_is_available() -> bool:
    """Return whether the configured read-only database has the serving schema."""
    con = None
    try:
        con = duckdb.connect(database=str(config.DB_FILE), read_only=True)
        for table_name, column_names in _REQUIRED_PREDICTION_COLUMNS.items():
            # Table and column names are selected only from the fixed allowlist above.
            columns = ", ".join(column_names)
            con.execute(f"SELECT {columns} FROM {table_name} LIMIT 0")
        return True
    except (duckdb.Error, OSError):
        logger.warning("Prediction database readiness check failed.", exc_info=True)
        return False
    finally:
        if con is not None:
            try:
                con.close()
            except duckdb.Error:
                logger.warning(
                    "Could not close readiness-check connection.", exc_info=True
                )


def list_inpatient_encounter_ids(limit: int = 20) -> list[str]:
    """Return recent inpatient encounter IDs for the Gradio picker."""
    if isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")

    con = None
    try:
        con = duckdb.connect(database=str(config.DB_FILE), read_only=True)
        rows = con.execute(
            """
            SELECT Id
            FROM encounters
            WHERE EncounterClass = 'IMP'
            ORDER BY Start DESC NULLS LAST, Id
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [str(row[0]) for row in rows]
    except (duckdb.Error, OSError):
        logger.info("No encounter presets are available from the prediction database.")
        return []
    finally:
        if con is not None:
            try:
                con.close()
            except duckdb.Error:
                logger.warning(
                    "Could not close encounter-list connection.", exc_info=True
                )


def engineer_features_for_encounter(
    encounter_id: str, con: duckdb.DuckDBPyConnection
) -> pd.DataFrame | None:
    """
    Fetches data and engineers all necessary features for a single encounter_id.

    Args:
        encounter_id (str): The unique identifier for the encounter.
        con (duckdb.DuckDBPyConnection): An active connection to the DuckDB database.

    Returns:
        pd.DataFrame: A single-row DataFrame with all the model features,
                      or None if the encounter is not found.
    """
    logger.info("Fetching base data for encounter_id: %s...", encounter_id)

    # --- 1. Fetch Base Admission and Demographic Data ---
    base_sql = """
    SELECT
        enc.Id AS encounter_id,
        p.Id AS patient_id,
        enc.Start AS admission_date,
        enc.Stop AS discharge_date,
        DATE_DIFF('day', enc.Start, enc.Stop) AS length_of_stay,
        DATE_DIFF('year', p.BirthDate, enc.Start) AS age_at_admission,
        p.Gender AS gender,
        p.Race AS race,
        p.Marital AS marital_status,
        enc.Description AS admission_reason,
        enc.Payer AS payer,
        enc.Total_Claim_Cost AS total_claim_cost,
        p.Income AS income,
        DAYNAME(enc.Start) AS admission_day_of_week,
        enc.ReasonCode AS primary_diagnosis_code,
        enc.Provider AS provider_id
    FROM encounters AS enc
    LEFT JOIN patients AS p ON enc.Patient = p.Id
    WHERE enc.Id = ? AND enc.EncounterClass = 'IMP'
    """
    features_df = con.execute(base_sql, [encounter_id]).fetchdf()
    if features_df.empty:
        logger.warning("No inpatient encounter found for ID: %s", encounter_id)
        return None

    logger.info("Base data fetched. Now engineering historical and clinical features.")

    # --- 2. Engineer High-Cardinality Interaction Feature ---
    features_df["payer_dx_interaction"] = (
        features_df["payer"].astype(str).fillna("unknown")
        + "_"
        + features_df["primary_diagnosis_code"].astype(str).fillna("unknown")
    )

    # --- 3. Engineer Historical Features (Prior Admissions) ---
    patient_id = features_df["patient_id"].iloc[0]
    admission_date = features_df["admission_date"].iloc[0]

    historical_sql = """
    SELECT COUNT(prior.Id) AS prior_admissions_last_year
    FROM encounters AS prior
    WHERE prior.Patient = ?
      AND prior.Start < ?
      AND DATE_DIFF('day', prior.Start, ?) <= 365
      AND prior.EncounterClass = 'IMP'
    """
    prior_admissions = con.execute(
        historical_sql, [patient_id, admission_date, admission_date]
    ).fetchone()[0]
    features_df["prior_admissions_last_year"] = prior_admissions

    # --- 4. Engineer Clinical Features (Counts) ---
    clinical_tables = {
        "diagnoses": "conditions",
        "procedures": "procedures",
        "medications": "medications",
    }
    for feature_name, table_name in clinical_tables.items():
        sql_clinical = f"""
        SELECT COUNT(Code) AS num_{feature_name}
        FROM {table_name}
        WHERE Encounter = ?
        """
        count = con.execute(sql_clinical, [encounter_id]).fetchone()[0]
        features_df[f"num_{feature_name}"] = count

    # --- 5. Final Cleanup ---
    # Drop intermediate columns not used in the model
    features_df = features_df.drop(
        columns=["patient_id", "admission_date", "discharge_date"]
    )

    return features_df


def make_prediction(encounter_id: str) -> dict:
    """
    Generates a readmission prediction for a given encounter ID.

    Args:
        encounter_id (str): The encounter ID to predict.

    Returns:
        dict: A dictionary containing the prediction probability, the binary
              prediction, and the threshold used.
    """
    logger.info("--- Starting Prediction Process ---")

    # --- 1. Load Model and Metadata ---
    logger.info("Loading model from: %s", config.MODEL_FILE)
    try:
        artifacts = load_model_artifacts(config.MODEL_FILE)
    except ModelArtifactsUnavailableError:
        logger.exception("Model artifacts are unavailable.")
        return _error_result("Model artifacts are unavailable.", MODEL_UNAVAILABLE)

    model = artifacts.model
    threshold = artifacts.threshold

    # --- 2. Fetch and Engineer Features ---
    con = None
    try:
        con = duckdb.connect(database=str(config.DB_FILE), read_only=True)
        features_df = engineer_features_for_encounter(encounter_id, con)
    except (duckdb.Error, OSError):
        logger.exception("Prediction database is unavailable.")
        return _error_result(
            "Prediction database is unavailable.", DATABASE_UNAVAILABLE
        )
    except Exception:
        logger.exception("Encounter feature generation failed unexpectedly.")
        return _error_result("Prediction could not be generated.", PREDICTION_FAILED)
    finally:
        if con is not None:
            try:
                con.close()
                logger.info("Database connection closed.")
            except duckdb.Error:
                logger.warning(
                    "Could not close prediction database connection.", exc_info=True
                )

    if features_df is None:
        return _error_result(
            f"Could not generate features for encounter ID {encounter_id}.",
            ENCOUNTER_NOT_FOUND,
        )

    try:
        logger.info("Generating prediction probability...")
        result = predict_feature_frame(model, threshold, features_df)
    except Exception:
        logger.exception("Prediction generation failed.")
        return _error_result("Prediction could not be generated.", PREDICTION_FAILED)

    logger.info(
        "Prediction complete. Probability: %.4f, Threshold: %s",
        result["readmission_probability"],
        threshold,
    )
    result = {"encounter_id": encounter_id, **result}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a readmission prediction for a single encounter."
    )
    parser.add_argument(
        "encounter_id",
        type=str,
        help="The encounter_id to generate a prediction for.",
    )
    args = parser.parse_args()

    # Example of how to run from the command line:
    # python -m src.predict "a933a39e-b98f-4171-8b9a-8a0a861d3e1d"
    prediction_result = make_prediction(args.encounter_id)
    print(json.dumps(prediction_result, indent=4))
