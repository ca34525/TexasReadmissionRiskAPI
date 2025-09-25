# src/predict.py

import argparse
import json
import logging
from pathlib import Path

import catboost as cb
import duckdb
import pandas as pd

# Import variables from our central configuration
from . import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
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
    logging.info(f"Fetching base data for encounter_id: {encounter_id}...")

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
    try:
        features_df = con.execute(base_sql, [encounter_id]).fetchdf()
        if features_df.empty:
            logging.warning(f"No inpatient encounter found for ID: {encounter_id}")
            return None
    except duckdb.Error as e:
        logging.error(f"Database error fetching base data: {e}")
        return None

    logging.info("Base data fetched. Now engineering historical and clinical features.")

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
    features_df = features_df.drop(columns=["patient_id", "admission_date", "discharge_date"])

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
    logging.info("--- Starting Prediction Process ---")
    
    # --- 1. Load Model and Metadata ---
    logging.info(f"Loading model from: {config.MODEL_FILE}")
    try:
        model = cb.CatBoostClassifier()
        model.load_model(str(config.MODEL_FILE))
        
        metadata_path = config.MODEL_FILE.parent / "model_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        threshold = metadata["optimal_threshold"]
    except FileNotFoundError as e:
        logging.error(f"Model or metadata file not found: {e}")
        return {"error": "Model artifacts not found."}

    # --- 2. Fetch and Engineer Features ---
    try:
        con = duckdb.connect(database=str(config.DB_FILE), read_only=True)
        features_df = engineer_features_for_encounter(encounter_id, con)
    finally:
        con.close() # Ensure connection is always closed
        logging.info("Database connection closed.")

    if features_df is None:
        return {"error": f"Could not generate features for encounter ID {encounter_id}."}

    # --- 3. Align Columns and Preprocess ---
    # Ensure the feature DataFrame has the same columns as the model was trained on
    model_features = model.feature_names_
    features_df = features_df.reindex(columns=model_features, fill_value=0)

    logging.info("Preprocessing categorical features for prediction...")
    for col in config.CATEGORICAL_FEATURES:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(str).fillna("missing")
            features_df[col] = features_df[col].astype("category")

    # --- 4. Generate Prediction ---
    logging.info("Generating prediction probability...")
    pred_proba = model.predict_proba(features_df)[0, 1]  # Probability of class 1
    prediction = 1 if pred_proba >= threshold else 0

    logging.info(f"Prediction complete. Probability: {pred_proba:.4f}, Threshold: {threshold}")

    result = {
        "encounter_id": encounter_id,
        "readmission_probability": float(pred_proba),
        "prediction": prediction,
        "threshold": threshold,
    }
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a readmission prediction for a single encounter.")
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