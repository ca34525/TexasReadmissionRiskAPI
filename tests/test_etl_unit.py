import re

import pytest

from src import etl
from src.etl import process_file


def test_process_file_reports_the_invalid_bundle_path(tmp_path):
    invalid_bundle = tmp_path / "invalid.json"
    invalid_bundle.write_text("not valid JSON", encoding="utf-8")

    with pytest.raises(RuntimeError, match=re.escape(str(invalid_bundle))):
        process_file(invalid_bundle)


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
