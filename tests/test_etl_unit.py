import json
import re

import pytest

from src import etl
from src.etl import process_file


def _write_bundle(tmp_path, entries):
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps({"resourceType": "Bundle", "entry": entries}),
        encoding="utf-8",
    )
    return bundle_path


def test_process_file_reports_the_invalid_bundle_path(tmp_path):
    invalid_bundle = tmp_path / "invalid.json"
    invalid_bundle.write_text("not valid JSON", encoding="utf-8")

    with pytest.raises(RuntimeError, match=re.escape(str(invalid_bundle))):
        process_file(invalid_bundle)


@pytest.mark.parametrize(
    ("full_url", "reference"),
    [
        ("urn:uuid:medication-entry-1", "urn:uuid:medication-entry-1"),
        (
            "https://example.test/fhir/Medication/medication-1",
            "Medication/medication-1",
        ),
        (
            "https://example.test/fhir/Medication/medication-1",
            "https://example.test/fhir/Medication/medication-1/_history/2",
        ),
        (None, "Medication/medication-1"),
    ],
)
def test_process_file_resolves_referenced_medication(tmp_path, full_url, reference):
    bundle_path = _write_bundle(
        tmp_path,
        [
            {
                "fullUrl": "urn:uuid:request-1",
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "request-1",
                    "medicationReference": {"reference": reference},
                },
            },
            {
                "fullUrl": full_url,
                "resource": {
                    "resourceType": "Medication",
                    "id": "medication-1",
                    "code": {
                        "coding": [
                            {
                                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                "code": "860975",
                                "display": "Metformin 500 MG Oral Tablet",
                            }
                        ]
                    },
                },
            },
        ],
    )

    medications = process_file(bundle_path)["medications"]

    assert len(medications) == 1
    assert medications[0]["Code"] == "860975"
    assert medications[0]["Description"] == "Metformin 500 MG Oral Tablet"
    assert sum(medication["Code"] is not None for medication in medications) == 1


@pytest.mark.parametrize("reference", ["#medication-1", "Medication/missing"])
def test_process_file_leaves_unresolved_medication_reference_uncoded(
    tmp_path, reference
):
    bundle_path = _write_bundle(
        tmp_path,
        [
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "request-1",
                    "medicationReference": {"reference": reference},
                }
            },
            {
                "resource": {
                    "resourceType": "Medication",
                    "id": "medication-1",
                    "code": {
                        "coding": [
                            {
                                "code": "860975",
                                "display": "Metformin 500 MG Oral Tablet",
                            }
                        ]
                    },
                }
            },
        ],
    )

    medication = process_file(bundle_path)["medications"][0]

    assert medication["Code"] is None
    assert medication["Description"] is None


def test_process_file_retains_direct_medication_code(tmp_path):
    bundle_path = _write_bundle(
        tmp_path,
        [
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "request-1",
                    "medicationCodeableConcept": {
                        "coding": [
                            {
                                "code": "313782",
                                "display": "Acetaminophen 325 MG Oral Tablet",
                            }
                        ]
                    },
                }
            }
        ],
    )

    medication = process_file(bundle_path)["medications"][0]

    assert medication["Code"] == "313782"
    assert medication["Description"] == "Acetaminophen 325 MG Oral Tablet"


def test_etl_fails_before_touching_database_when_no_bundles_exist(
    tmp_path, monkeypatch
):
    empty_fhir_directory = tmp_path / "fhir"
    empty_fhir_directory.mkdir()
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    database_path = output_directory / "existing.duckdb"
    database_path.write_bytes(b"existing database placeholder")

    monkeypatch.setattr(etl.config, "FHIR_DIR", empty_fhir_directory)
    monkeypatch.setattr(etl.config, "OUTPUT_DIR", output_directory)
    monkeypatch.setattr(etl.config, "DB_FILE", database_path)
    monkeypatch.setattr(etl.config, "MAX_FILES_TO_PROCESS", None)

    with pytest.raises(FileNotFoundError, match="No FHIR JSON bundles"):
        etl.main()

    assert database_path.read_bytes() == b"existing database placeholder"
