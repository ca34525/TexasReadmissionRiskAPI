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
                ('patient-1', '1980-06-15', 'female', 'white', 'M', 60000),
                ('patient-2', '1975-03-20', 'male', 'black', 'S', 45000),
                ('patient-3', '1965-09-08', 'female', 'asian', 'M', 70000),
                ('patient-4', '1990-11-12', 'male', 'white', 'S', 55000),
                ('patient-5', '1955-02-03', 'female', 'black', 'W', 65000);

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
                 'Z09', 'provider-1'),
                ('overlap-index', 'patient-2', '2025-01-01', '2025-01-10', 'IMP',
                 'Inpatient admission', 'Index admission', 'Medicare', 11000,
                 'J18.9', 'provider-2'),
                ('overlap-stay', 'patient-2', '2025-01-05', '2025-01-20', 'IMP',
                 'Inpatient admission', 'Overlapping stay', 'Medicare', 14000,
                 'I50.9', 'provider-2'),
                ('post-discharge', 'patient-2', '2025-01-25', '2025-01-30', 'IMP',
                 'Inpatient admission', 'Post-discharge admission', 'Medicare',
                 12500, 'E11.9', 'provider-2'),
                ('boundary-index', 'patient-3', '2025-04-01 08:00:00',
                 '2025-04-10 12:00:00', 'IMP', 'Inpatient admission',
                 'Index admission', 'Aetna', 15000, 'J18.9', 'provider-3'),
                ('boundary-transfer', 'patient-3', '2025-04-10 12:00:00',
                 '2025-04-15 10:00:00', 'IMP', 'Inpatient admission',
                 'Same-time transfer', 'Aetna', 10000, 'Z09', 'provider-3'),
                ('overlap-only-index', 'patient-4', '2025-06-01', '2025-06-10',
                 'IMP', 'Inpatient admission', 'Index admission', 'Aetna', 8000,
                 'J18.9', 'provider-4'),
                ('overlap-only-stay', 'patient-4', '2025-06-05', '2025-06-15',
                 'IMP', 'Inpatient admission', 'Overlapping stay', 'Aetna', 9000,
                 'I50.9', 'provider-4'),
                ('day-30-index', 'patient-5', '2025-08-01 12:00:00',
                 '2025-08-02 12:00:00', 'IMP', 'Inpatient admission',
                 'Index admission', 'Medicare', 7000, 'J18.9', 'provider-5'),
                ('day-30-readmission', 'patient-5', '2025-09-01 12:00:00',
                 '2025-09-02 12:00:00', 'IMP', 'Inpatient admission',
                 '30-day readmission', 'Medicare', 7500, 'I50.9', 'provider-5'),
                ('day-31-readmission', 'patient-5', '2025-10-03 12:00:00',
                 '2025-10-04 12:00:00', 'IMP', 'Inpatient admission',
                 '31-day readmission', 'Medicare', 7800, 'E11.9', 'provider-5');

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


def test_create_features_uses_next_admission_after_discharge(tmp_path):
    database_path = tmp_path / "synthea.duckdb"
    output_path = tmp_path / "features.parquet"
    _create_source_database(database_path)

    create_features(database_path, output_path)

    features = pd.read_parquet(output_path).set_index("encounter_id")

    # The intervening stay starts before the index discharge, so the later
    # admission is the qualifying readmission for the index encounter.
    assert features.loc["overlap-index", config.TARGET_VARIABLE] == 1
    assert features.loc["overlap-stay", config.TARGET_VARIABLE] == 1
    assert features.loc["post-discharge", config.TARGET_VARIABLE] == 0
    # Overlapping encounters alone do not create a post-discharge readmission.
    assert features.loc["overlap-only-index", config.TARGET_VARIABLE] == 0
    assert features.loc["overlap-only-stay", config.TARGET_VARIABLE] == 0
    # A same-instant inpatient transition is not a new admission after discharge.
    assert features.loc["boundary-index", config.TARGET_VARIABLE] == 0
    assert features.loc["boundary-transfer", config.TARGET_VARIABLE] == 0
    # Preserve the established calendar-day definition of the 30-day window.
    assert features.loc["day-30-index", config.TARGET_VARIABLE] == 1
    assert features.loc["day-30-readmission", config.TARGET_VARIABLE] == 0
    assert features.loc["day-31-readmission", config.TARGET_VARIABLE] == 0
