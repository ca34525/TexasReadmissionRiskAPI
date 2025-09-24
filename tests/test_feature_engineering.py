import duckdb
import pandas as pd
import pytest
from src.feature_engineering import create_features


@pytest.fixture(scope="module")
def setup_test_db(tmp_path_factory):
    """
    Creates a temporary DuckDB database with controlled test data
    for feature engineering validation.
    """
    db_file = tmp_path_factory.mktemp("test_db") / "test_fe.duckdb"
    con = duckdb.connect(database=str(db_file), read_only=False)

    # --- Create Tables ---
    con.execute("CREATE TABLE patients (Id VARCHAR, BirthDate DATE, Gender VARCHAR, Race VARCHAR, Marital VARCHAR, Income INTEGER);")
    con.execute("CREATE TABLE encounters (Id VARCHAR, Start TIMESTAMP, Stop TIMESTAMP, Patient VARCHAR, Provider VARCHAR, Payer VARCHAR, EncounterClass VARCHAR, Code VARCHAR, Description VARCHAR, Total_Claim_Cost FLOAT, ReasonCode VARCHAR, ReasonDescription VARCHAR);")
    con.execute("CREATE TABLE conditions (Encounter VARCHAR, Code VARCHAR);")
    con.execute("CREATE TABLE procedures (Encounter VARCHAR, Code VARCHAR);")
    con.execute("CREATE TABLE medications (Encounter VARCHAR, Code VARCHAR);")

    # --- Insert Data ---
    con.execute("INSERT INTO patients VALUES ('patient-1', '1980-01-01', 'male', 'white', 'M', 50000);")
    # CORRECTED THE DATE HERE to be within 365 days of the index admission
    con.execute("INSERT INTO encounters VALUES ('enc-prior', '2024-06-01 10:00:00', '2024-06-03 12:00:00', 'patient-1', 'prov-1', 'Medicare', 'IMP', '123', 'Reason 1', 1000.0, 'R1', 'Desc 1');")
    con.execute("INSERT INTO encounters VALUES ('enc-1a', '2025-01-10 10:00:00', '2025-01-15 12:00:00', 'patient-1', 'prov-1', 'Medicare', 'IMP', '123', 'Reason 1', 2000.0, 'R1', 'Desc 1');")
    con.execute("INSERT INTO encounters VALUES ('enc-1b', '2025-02-01 10:00:00', '2025-02-03 12:00:00', 'patient-1', 'prov-1', 'Medicare', 'IMP', '456', 'Reason 2', 1500.0, 'R2', 'Desc 2');")
    con.execute("INSERT INTO conditions VALUES ('enc-1a', 'C1'), ('enc-1a', 'C2');")
    con.execute("INSERT INTO procedures VALUES ('enc-1a', 'P1');")

    con.execute("INSERT INTO patients VALUES ('patient-2', '1990-05-15', 'female', 'black', 'S', 75000);")
    con.execute("INSERT INTO encounters VALUES ('enc-2a', '2025-03-01 08:00:00', '2025-03-05 18:00:00', 'patient-2', 'prov-2', 'Anthem', 'IMP', '789', 'Reason 3', 5000.0, 'R3', 'Desc 3');")
    con.execute("INSERT INTO encounters VALUES ('enc-2b', '2025-05-01 08:00:00', '2025-05-02 18:00:00', 'patient-2', 'prov-2', 'Anthem', 'IMP', '101', 'Reason 4', 800.0, 'R4', 'Desc 4');")

    con.execute("INSERT INTO patients VALUES ('patient-3', '1975-11-20', 'male', 'asian', 'M', 120000);")
    con.execute("INSERT INTO encounters VALUES ('enc-3', '2025-06-01 09:00:00', '2025-06-10 17:00:00', 'patient-3', 'prov-1', 'NO_INSURANCE', 'IMP', '112', 'Reason 5', 12000.0, 'R5', 'Desc 5');")
    con.execute("INSERT INTO medications VALUES ('enc-3', 'M1'), ('enc-3', 'M2'), ('enc-3', 'M3');")

    con.close()
    return db_file


def test_create_features(setup_test_db, tmp_path):
    """
    Tests the create_features function against a controlled dataset.
    """
    db_path = setup_test_db
    output_path = tmp_path / "test_features.parquet"
    create_features(db_path, output_path)

    assert output_path.exists(), "Output parquet file was not created."

    df = pd.read_parquet(output_path)
    assert len(df) == 6, "Expected 6 inpatient encounters in the output."

    df = df.sort_values(by="encounter_id").reset_index(drop=True)

    p1_admission = df[df["encounter_id"] == "enc-1a"].iloc[0]
    assert p1_admission["readmitted_within_30_days"] == 1
    assert p1_admission["length_of_stay"] == 5
    assert p1_admission["age_at_admission"] == 45
    assert p1_admission["prior_admissions_last_year"] == 1.0
    assert p1_admission["num_diagnoses"] == 2.0
    assert p1_admission["num_procedures"] == 1.0
    assert p1_admission["num_medications"] == 0.0

    p2_admission = df[df["encounter_id"] == "enc-2a"].iloc[0]
    assert p2_admission["readmitted_within_30_days"] == 0
    assert p2_admission["length_of_stay"] == 4
    assert p2_admission["prior_admissions_last_year"] == 0.0

    p3_admission = df[df["encounter_id"] == "enc-3"].iloc[0]
    assert p3_admission["readmitted_within_30_days"] == 0
    assert p3_admission["num_medications"] == 3.0
    assert p3_admission["payer_dx_interaction"] == "NO_INSURANCE_R5"