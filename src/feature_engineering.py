import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def create_features(db_path: Path, output_path: Path):
    """
    Connects to a DuckDB database, engineers features for the readmission
    prediction model, and saves the final dataset.

    Args:
        db_path (Path): Path to the source DuckDB database file.
        output_path (Path): Path to save the output Parquet file.
    """
    logging.info(f"Connecting to DuckDB database: {db_path}")
    con = duckdb.connect(database=str(db_path), read_only=True)

    # --- 1. Engineer the Target Variable (readmitted_within_30_days) ---
    logging.info("Step 1: Engineering the target variable...")
    sql_target = """
    WITH PatientAdmissions AS (
        SELECT
            Id AS encounter_id,
            Patient AS patient_id,
            Start AS admission_date,
            Stop AS discharge_date,
            LEAD(Start, 1) OVER(
                PARTITION BY Patient ORDER BY Start
            ) AS next_admission_date
        FROM encounters
        WHERE EncounterClass = 'IMP'
    )
    SELECT
        encounter_id,
        patient_id,
        admission_date,
        discharge_date,
        next_admission_date,
        DATE_DIFF('day', discharge_date, next_admission_date)
            AS days_to_next_admission,
        CASE
            WHEN DATE_DIFF('day', discharge_date, next_admission_date) <= 30
            THEN 1
            ELSE 0
        END AS readmitted_within_30_days
    FROM PatientAdmissions
    WHERE discharge_date IS NOT NULL
    """
    readmissions_df = con.execute(sql_target).fetchdf()
    logging.info(f"Identified {len(readmissions_df)} index admissions.")

    # --- 2. Add Demographics and Admission-Level Features ---
    logging.info("Step 2: Adding demographic and admission features...")
    sql_demographics = """
    SELECT
        readmissions.encounter_id,
        readmissions.patient_id,
        readmissions.readmitted_within_30_days,
        readmissions.admission_date,
        readmissions.discharge_date,
        readmissions.next_admission_date,
        readmissions.days_to_next_admission,
        DATE_DIFF('day', readmissions.admission_date, readmissions.discharge_date)
            AS length_of_stay,
        DATE_DIFF('year', p.BirthDate, readmissions.admission_date)
            AS age_at_admission,
        p.Gender AS gender,
        p.Race AS race,
        p.Marital AS marital_status,
        enc.Description AS admission_reason,
        enc.ReasonDescription AS admission_reason_detail,
        enc.Payer AS payer,
        enc.Total_Claim_Cost AS total_claim_cost,
        p.Income AS income,
        DAYNAME(readmissions.admission_date) AS admission_day_of_week,
        enc.ReasonCode AS primary_diagnosis_code,
        enc.Provider AS provider_id
    FROM readmissions_df AS readmissions
    LEFT JOIN patients AS p ON readmissions.patient_id = p.Id
    LEFT JOIN encounters AS enc ON readmissions.encounter_id = enc.Id
    """
    model_df = con.execute(sql_demographics).fetchdf()
    logging.info("Successfully joined demographic and admission data.")

    # --- 3. Engineer High-Cardinality Interaction Feature ---
    logging.info("Step 3: Engineering interaction features...")
    model_df["payer_dx_interaction"] = (
        model_df["payer"].astype(str).fillna("unknown")
        + "_"
        + model_df["primary_diagnosis_code"].astype(str).fillna("unknown")
    )

    # --- 4. Engineer Historical Features ---
    logging.info("Step 4: Engineering historical features...")
    sql_historical = """
    SELECT
        index_admission.Id AS encounter_id,
        COUNT(prior_admissions.Id) AS prior_admissions_last_year
    FROM encounters AS index_admission
    LEFT JOIN encounters AS prior_admissions
        ON index_admission.Patient = prior_admissions.Patient
        AND prior_admissions.Start < index_admission.Start
        AND DATE_DIFF('day', prior_admissions.Start, index_admission.Start) <= 365
        AND prior_admissions.EncounterClass = 'IMP'
    WHERE index_admission.EncounterClass = 'IMP'
    GROUP BY index_admission.Id
    """
    prior_admissions_df = con.execute(sql_historical).fetchdf()
    model_df = pd.merge(model_df, prior_admissions_df, on="encounter_id", how="left")
    model_df["prior_admissions_last_year"] = model_df[
        "prior_admissions_last_year"
    ].fillna(0)
    logging.info("Successfully added prior admissions count.")

    # --- 5. Engineer Clinical Features (Counts) ---
    logging.info("Step 5: Engineering clinical features (counts)...")
    clinical_tables = {
        "diagnoses": "conditions",
        "procedures": "procedures",
        "medications": "medications",
    }
    for feature_name, table_name in clinical_tables.items():
        logging.info(f"  - Counting from table: {table_name}")
        sql_clinical = f"""
        SELECT
            Encounter AS encounter_id,
            COUNT(Code) AS num_{feature_name}
        FROM {table_name}
        GROUP BY Encounter
        """
        clinical_df = con.execute(sql_clinical).fetchdf()
        model_df = pd.merge(model_df, clinical_df, on="encounter_id", how="left")
        model_df[f"num_{feature_name}"] = model_df[f"num_{feature_name}"].fillna(0)

    # --- 6. Final Cleanup ---
    logging.info("Step 6: Cleaning up columns before saving...")
    # These columns were intermediate and are not features for the model
    columns_to_drop = [
        "patient_id",
        "admission_date",
        "discharge_date",
        "next_admission_date",
        "days_to_next_admission",
        "admission_reason_detail",
    ]
    model_df = model_df.drop(columns=columns_to_drop)

    # --- 7. Save Final Dataset & Cleanup ---
    logging.info("Step 7: Saving final dataset and cleaning up...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_parquet(output_path, index=False)
    con.close()
    logging.info(f"Successfully saved feature table to {output_path}")
    logging.info(
        f"Final dataset has {len(model_df):,} rows and {len(model_df.columns)} columns."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the feature engineering pipeline."
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default=Path("output/synthea_fhir.duckdb"),
        help="Path to the source DuckDB database file.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("output/readmissions_dataset.parquet"),
        help="Path to save the final feature dataset (Parquet).",
    )
    args = parser.parse_args()

    create_features(db_path=args.db_path, output_path=args.output_path)
