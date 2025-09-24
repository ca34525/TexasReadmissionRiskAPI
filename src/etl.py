# src/etl.py

"""
Main ETL script for processing Synthea FHIR data into a DuckDB database.

This script performs the following steps:
1.  Connects to a DuckDB database, clearing and recreating tables.
2.  Scans a directory for FHIR JSON bundle files.
3.  Processes the files in parallel using multiple CPU cores.
4.  For each file, it parses Patient, Encounter, Condition, Procedure,
    and other relevant resources into a clean, flat structure.
5.  Enriches records by linking financial data (from ExplanationOfBenefit)
    and social determinants of health data (from specific Observations).
6.  Inserts the processed data into the corresponding DuckDB tables in batches.
"""

import multiprocessing
from pathlib import Path

import duckdb
import orjson
import pandas as pd
from tqdm import tqdm

# Import modules from our source directory
from . import config, utils


# --- 1. RESOURCE PARSING FUNCTIONS ---
# Each function transforms a specific FHIR resource into a flat dictionary.


def parse_patient(resource: dict) -> dict:
    """Parses a FHIR Patient resource with robust error handling."""
    primary_name = resource.get("name", [{}])[0]
    maiden_name_obj = next(
        (name for name in resource.get("name", []) if name.get("use") == "maiden"),
        None,
    )
    address = utils.get_first_address(resource) or {}

    return {
        "Id": resource.get("id"),
        "BirthDate": resource.get("birthDate"),
        "DeathDate": resource.get("deceasedDateTime"),
        "SSN": utils.get_identifier(resource, "http://hl7.org/fhir/sid/us-ssn"),
        "Drivers": utils.get_identifier(
            resource, "urn:oid:2.16.840.1.113883.4.3.25"
        ),
        "Passport": utils.get_identifier(
            resource, "http://standardhealthrecord.org/fhir/sid/passport-number"
        ),
        "Prefix": next(iter(primary_name.get("prefix", [])), None),
        "First": next(iter(primary_name.get("given", [])), None),
        "Middle": (
            primary_name.get("given")[1]
            if len(primary_name.get("given", [])) > 1
            else None
        ),
        "Last": primary_name.get("family"),
        "Suffix": next(iter(primary_name.get("suffix", [])), None),
        "Maiden": maiden_name_obj.get("family") if maiden_name_obj else None,
        "Marital": utils.get_coding(resource.get("maritalStatus")).get("code"),
        "Race": utils.get_us_core_extension_value(
            resource, "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
        ),
        "Ethnicity": utils.get_us_core_extension_value(
            resource,
            "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
        ),
        "Gender": resource.get("gender"),
        "BirthPlace": address.get("city"),
        "Address": next(iter(address.get("line", [])), None),
        "City": address.get("city"),
        "State": address.get("state"),
        "County": utils.get_extension_value(
            address, "http://hl7.org/fhir/StructureDefinition/iso21090-ADXP-county"
        ),
        "FIPS": utils.get_extension_value(
            address, "http://synthetichealth.github.io/synthea/fips-county-code"
        ),
        "Zip": address.get("postalCode"),
        "Lat": None,  # Placeholder, geo extension is complex
        "Lon": None,  # Placeholder, geo extension is complex
        "Healthcare_Expenses": utils.get_extension_value(
            resource,
            "http://synthetichealth.github.io/synthea/financial-information#healthcare-expenses",
        ),
        "Healthcare_Coverage": utils.get_extension_value(
            resource,
            "http://synthetichealth.github.io/synthea/financial-information#healthcare-coverage",
        ),
        "Income": None,
    }


def parse_sdoh_observation(resource: dict) -> dict | None:
    """
    Parses a FHIR Observation resource specifically to find the PRAPARE
    survey and extract the patient's income from its components.
    """
    if utils.get_coding(resource.get("code")).get("code") != "93025-5":
        return None

    patient_id = utils.get_clean_id(resource.get("subject"))
    income = None

    for component in resource.get("component", []):
        if utils.get_coding(component.get("code")).get("code") == "63586-2":
            if value_quantity := component.get("valueQuantity"):
                income = value_quantity.get("value")
                break

    if patient_id and income is not None:
        return {"Patient": patient_id, "Income": income}

    return None


def parse_condition(resource: dict) -> dict:
    """Parses a FHIR Condition resource."""
    code_info = utils.get_coding(resource.get("code"))
    return {
        "Start": resource.get("onsetDateTime"),
        "Stop": resource.get("abatementDateTime"),
        "Patient": utils.get_clean_id(resource.get("subject")),
        "Encounter": utils.get_clean_id(resource.get("encounter")),
        "System": code_info.get("system"),
        "Code": code_info.get("code"),
        "Description": code_info.get("display"),
    }


def parse_procedure(resource: dict) -> dict:
    """Parses a FHIR Procedure resource."""
    code_info = utils.get_coding(resource.get("code"))
    reason_info = utils.get_coding(
        next(iter(resource.get("reasonCode", [])), None)
    )
    return {
        "Start": resource.get("performedPeriod", {}).get("start"),
        "Stop": resource.get("performedPeriod", {}).get("end"),
        "Patient": utils.get_clean_id(resource.get("subject")),
        "Encounter": utils.get_clean_id(resource.get("encounter")),
        "System": code_info.get("system"),
        "Code": code_info.get("code"),
        "Description": code_info.get("display"),
        "Base_Cost": utils.get_extension_value(
            resource,
            "http://synthetichealth.github.io/synthea/financial-information#base-cost",
        ),
        "ReasonCode": reason_info.get("code"),
        "ReasonDescription": reason_info.get("display"),
    }


def parse_medication(resource: dict) -> dict:
    """Parses a FHIR MedicationRequest resource."""
    code_info = utils.get_coding(resource.get("medicationCodeableConcept"))
    reason_info = utils.get_coding(
        next(iter(resource.get("reasonCode", [])), None)
    )
    return {
        "Start": resource.get("authoredOn"),
        "Stop": (
            resource.get("dispenseRequest", {})
            .get("validityPeriod", {})
            .get("end")
        ),
        "Patient": utils.get_clean_id(resource.get("subject")),
        "Payer": None,
        "Encounter": utils.get_clean_id(resource.get("encounter")),
        "Code": code_info.get("code"),
        "Description": code_info.get("display"),
        "Base_Cost": utils.get_extension_value(
            resource,
            "http://synthetichealth.github.io/synthea/financial-information#base-cost",
        ),
        "Payer_Coverage": utils.get_extension_value(
            resource,
            "http://synthetichealth.github.io/synthea/financial-information#payer-coverage",
        ),
        "Dispenses": 1
        + resource.get("dispenseRequest", {}).get("numberOfRepeatsAllowed", 0),
        "TotalCost": utils.get_extension_value(
            resource,
            "http://synthetichealth.github.io/synthea/financial-information#total-cost",
        ),
        "ReasonCode": reason_info.get("code"),
        "ReasonDescription": reason_info.get("display"),
    }


def parse_encounter(resource: dict) -> dict:
    """Parses a FHIR Encounter resource."""
    primary_type_coding = utils.get_coding(
        next(iter(resource.get("type", [])), None)
    )
    primary_reason_coding = utils.get_coding(
        next(iter(resource.get("reasonCode", [])), None)
    )
    return {
        "Id": resource.get("id"),
        "Start": resource.get("period", {}).get("start"),
        "Stop": resource.get("period", {}).get("end"),
        "Patient": utils.get_clean_id(resource.get("subject")),
        "Organization": utils.get_clean_id(resource.get("serviceProvider")),
        "Provider": utils.get_clean_id(
            next(iter(resource.get("participant", [])), {}).get("individual")
        ),
        "Payer": None,
        "EncounterClass": resource.get("class", {}).get("code"),
        "Code": primary_type_coding.get("code"),
        "Description": primary_type_coding.get("display"),
        "Base_Encounter_Cost": utils.get_extension_value(
            resource,
            "http://synthetichealth.github.io/synthea/financial-information#base-encounter-cost",
        ),
        "Total_Claim_Cost": None,
        "Payer_Coverage": utils.get_extension_value(
            resource,
            "http://synthetichealth.github.io/synthea/financial-information#payer-coverage",
        ),
        "ReasonCode": primary_reason_coding.get("code"),
        "ReasonDescription": primary_reason_coding.get("display"),
    }


def parse_explanation_of_benefit(resource: dict) -> dict:
    """Parses a FHIR ExplanationOfBenefit resource to get financial data."""
    total_cost_money = next(iter(resource.get("total", [])), {}).get("amount", {})

    encounter_ref = None
    if encounters := resource.get("encounter"):
        encounter_ref = encounters[0]
    elif items := resource.get("item"):
        if item_encounters := items[0].get("encounter"):
            encounter_ref = item_encounters[0]

    return {
        "Encounter": utils.get_clean_id(encounter_ref),
        "Payer": resource.get("insurer", {}).get("display"),
        "Total_Claim_Cost": total_cost_money.get("value"),
    }


# --- 2. MULTIPROCESSING WORKER FUNCTION ---


def process_file(file_path: Path) -> dict:
    """
    Processes a single FHIR JSON file, parsing all relevant resources
    and enriching encounters with EOB data and patients with SDOH data.
    """
    data = {
        "patients": [],
        "encounters": [],
        "conditions": [],
        "procedures": [],
        "medications": [],
    }
    eobs_raw = []
    sdoh_obs_raw = []

    try:
        with open(file_path, "rb") as f:
            bundle = orjson.loads(f.read())

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if not resource:
                continue

            resource_type = resource.get("resourceType")
            if resource_type == "Patient":
                data["patients"].append(parse_patient(resource))
            elif resource_type == "Encounter":
                data["encounters"].append(parse_encounter(resource))
            elif resource_type == "Condition":
                data["conditions"].append(parse_condition(resource))
            elif resource_type == "Procedure":
                data["procedures"].append(parse_procedure(resource))
            elif resource_type == "MedicationRequest":
                data["medications"].append(parse_medication(resource))
            elif resource_type == "ExplanationOfBenefit":
                eobs_raw.append(parse_explanation_of_benefit(resource))
            elif resource_type == "Observation":
                if sdoh_data := parse_sdoh_observation(resource):
                    sdoh_obs_raw.append(sdoh_data)

        # Enrich Patients with Income
        if sdoh_obs_raw and data["patients"]:
            income_lookup = {
                obs["Patient"]: obs["Income"] for obs in sdoh_obs_raw if obs
            }
            for patient in data["patients"]:
                if income := income_lookup.get(patient["Id"]):
                    patient["Income"] = income

        # Enrich Encounters with EOB data
        if eobs_raw and data["encounters"]:
            eob_lookup = {
                eob["Encounter"]: eob
                for eob in eobs_raw
                if eob and eob.get("Encounter")
            }
            for encounter in data["encounters"]:
                if eob_data := eob_lookup.get(encounter["Id"]):
                    encounter["Payer"] = eob_data.get("Payer")
                    encounter["Total_Claim_Cost"] = eob_data.get(
                        "Total_Claim_Cost"
                    )
    except Exception:
        pass  # Silently ignore files that fail to parse

    return data


# --- 3. DATABASE INITIALIZATION ---


def initialize_database(con: duckdb.DuckDBPyConnection):
    """Drops and recreates all tables in the database."""
    print("Dropping existing tables...")
    con.execute("DROP TABLE IF EXISTS patients;")
    con.execute("DROP TABLE IF EXISTS encounters;")
    con.execute("DROP TABLE IF EXISTS conditions;")
    con.execute("DROP TABLE IF EXISTS procedures;")
    con.execute("DROP TABLE IF EXISTS medications;")

    print("Creating new tables...")
    con.execute(
        """
        CREATE TABLE patients (
            Id VARCHAR PRIMARY KEY, BirthDate DATE, DeathDate DATE, SSN VARCHAR,
            Drivers VARCHAR, Passport VARCHAR, Prefix VARCHAR, First VARCHAR,
            Middle VARCHAR, Last VARCHAR, Suffix VARCHAR, Maiden VARCHAR,
            Marital VARCHAR(1), Race VARCHAR, Ethnicity VARCHAR, Gender VARCHAR,
            BirthPlace VARCHAR, Address VARCHAR, City VARCHAR, State VARCHAR,
            County VARCHAR, FIPS VARCHAR, Zip VARCHAR, Lat FLOAT, Lon FLOAT,
            Healthcare_Expenses FLOAT, Healthcare_Coverage FLOAT, Income BIGINT
        );
    """
    )
    con.execute(
        """
        CREATE TABLE conditions (
            Start DATE, Stop DATE, Patient VARCHAR, Encounter VARCHAR,
            System VARCHAR, Code VARCHAR, Description VARCHAR
        );
    """
    )
    con.execute(
        """
        CREATE TABLE procedures (
            Start TIMESTAMP, Stop TIMESTAMP, Patient VARCHAR, Encounter VARCHAR,
            System VARCHAR, Code VARCHAR, Description VARCHAR, Base_Cost FLOAT,
            ReasonCode VARCHAR, ReasonDescription VARCHAR
        );
    """
    )
    con.execute(
        """
        CREATE TABLE medications (
            Start TIMESTAMP, Stop TIMESTAMP, Patient VARCHAR, Payer VARCHAR,
            Encounter VARCHAR, Code VARCHAR, Description VARCHAR, Base_Cost FLOAT,
            Payer_Coverage FLOAT, Dispenses INTEGER, TotalCost FLOAT,
            ReasonCode VARCHAR, ReasonDescription VARCHAR
        );
    """
    )
    con.execute(
        """
        CREATE TABLE encounters (
            Id VARCHAR PRIMARY KEY, Start TIMESTAMP, Stop TIMESTAMP, Patient VARCHAR,
            Organization VARCHAR, Provider VARCHAR, Payer VARCHAR,
            EncounterClass VARCHAR, Code VARCHAR, Description VARCHAR,
            Base_Encounter_Cost FLOAT, Total_Claim_Cost FLOAT,
            Payer_Coverage FLOAT, ReasonCode VARCHAR, ReasonDescription VARCHAR
        );
    """
    )
    print("✅ All tables created successfully.")


# --- 4. MAIN EXECUTION SCRIPT ---


def main():
    """Main function to run the entire ETL pipeline."""
    # Ensure output directory exists
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Connecting to DuckDB database: {config.DB_FILE}")
    con = duckdb.connect(str(config.DB_FILE), read_only=False)
    initialize_database(con)

    print("Finding all FHIR JSON bundles...")
    fhir_files = list(config.FHIR_DIR.rglob("*.json"))
    if config.MAX_FILES_TO_PROCESS:
        fhir_files = fhir_files[: config.MAX_FILES_TO_PROCESS]
    total_files = len(fhir_files)
    print(f"Found {total_files:,} files to process.")

    file_chunks = [
        fhir_files[i : i + config.BATCH_SIZE]
        for i in range(0, total_files, config.BATCH_SIZE)
    ]
    num_chunks = len(file_chunks)
    print(
        f"Processing in {num_chunks} batches of up to {config.BATCH_SIZE} files."
    )

    print(f"Using {config.CPU_COUNT} worker processes.")

    for i, chunk in enumerate(file_chunks):
        print(f"\n--- Processing Batch {i+1}/{num_chunks} ({len(chunk)} files) ---")

        with multiprocessing.Pool(processes=config.CPU_COUNT) as pool:
            results = list(
                tqdm(pool.imap_unordered(process_file, chunk), total=len(chunk))
            )

        batch_data = {
            "patients": [
                item for res in results for item in res.get("patients", [])
            ],
            "encounters": [
                item for res in results for item in res.get("encounters", [])
            ],
            "conditions": [
                item for res in results for item in res.get("conditions", [])
            ],
            "procedures": [
                item for res in results for item in res.get("procedures", [])
            ],
            "medications": [
                item for res in results for item in res.get("medications", [])
            ],
        }

        for name, data_list in batch_data.items():
            if data_list:
                print(f"  Inserting {len(data_list):,} records into '{name}'...")
                df = pd.DataFrame(data_list)
                con.append(name, df)
            else:
                print(f"  No data for '{name}' in this batch.")

        print(f"✅ Batch {i+1}/{num_chunks} complete.")

    print("\n🎉 All batches processed successfully!")
    con.close()
    print("Database connection closed.")


if __name__ == "__main__":
    main()