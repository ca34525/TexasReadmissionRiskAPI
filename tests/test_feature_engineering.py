import duckdb
import pandas as pd

from src import config
from src.feature_engineering import create_features


def _create_source_database(database_path):
    con = duckdb.connect(str(database_path))
    try:
        con.execute(
            """
            CREATE TABLE patients (
                Id VARCHAR, BirthDate DATE, Gender VARCHAR, Race VARCHAR,
                Marital VARCHAR, Income BIGINT
            );
            INSERT INTO patients VALUES
                ('patient-1', '1980-06-15', 'female', 'white', 'M', 60000);

            CREATE TABLE encounters (
                Id VARCHAR, Patient VARCHAR, Start TIMESTAMP, Stop TIMESTAMP,
                EncounterClass VARCHAR, Description VARCHAR,
                ReasonDescription VARCHAR, Payer VARCHAR,
                Total_Claim_Cost DOUBLE, ReasonCode VARCHAR, Provider VARCHAR
            );
            INSERT INTO encounters VALUES
                ('enc-1', 'patient-1', '2025-01-01', '2025-01-05', 'IMP',
                 'Inpatient admission', 'Pneumonia', 'Medicare', 12000,
                 'J18.9', 'provider-1'),
                ('enc-ambulatory', 'patient-1', '2025-01-10', '2025-01-10',
                 'AMB', 'Office visit', NULL, 'Medicare', 200,
                 NULL, 'provider-1'),
                ('enc-2', 'patient-1', '2025-01-25', '2025-01-30', 'IMP',
                 'Inpatient admission', 'Heart failure', 'Aetna', 18000,
                 'I50.9', 'provider-1'),
                ('enc-3', 'patient-1', '2025-03-15', '2025-03-18', 'IMP',
                 'Inpatient admission', 'Follow-up', 'Aetna', 9000,
                 'Z09', 'provider-1');

            CREATE TABLE conditions (Encounter VARCHAR, Code VARCHAR);
            INSERT INTO conditions VALUES
                ('enc-1', 'J18.9'), ('enc-1', 'E11.9');

            CREATE TABLE procedures (Encounter VARCHAR, Code VARCHAR);
            INSERT INTO procedures VALUES ('enc-1', 'procedure-1');

            CREATE TABLE medications (Encounter VARCHAR, Code VARCHAR);
            INSERT INTO medications VALUES ('enc-2', 'medication-1');
            """
        )
    finally:
        con.close()


def test_create_features_builds_target_and_model_columns(tmp_path):
    database_path = tmp_path / "synthea.duckdb"
    output_path = tmp_path / "features.parquet"
    _create_source_database(database_path)

    create_features(database_path, output_path)

    features = pd.read_parquet(output_path).set_index("encounter_id")
    expected_columns = set(config.CATEGORICAL_FEATURES) | {
        "length_of_stay",
        "age_at_admission",
        "total_claim_cost",
        "income",
        "prior_admissions_last_year",
        "num_diagnoses",
        "num_procedures",
        "num_medications",
        config.TARGET_VARIABLE,
    }

    assert set(features.columns) == expected_columns
    assert features.loc["enc-1", config.TARGET_VARIABLE] == 1
    assert features.loc["enc-2", config.TARGET_VARIABLE] == 0
    assert features.loc["enc-3", config.TARGET_VARIABLE] == 0
    assert features.loc["enc-1", "payer_dx_interaction"] == "Medicare_J18.9"
    assert features.loc["enc-2", "prior_admissions_last_year"] == 1
    assert features.loc["enc-3", "prior_admissions_last_year"] == 2
    assert features.loc["enc-1", "num_diagnoses"] == 2
    assert features.loc["enc-1", "num_procedures"] == 1
    assert features.loc["enc-2", "num_medications"] == 1
