# tests/test_etl.py

"""
Tests for verifying the integrity of the DuckDB database after the ETL process.
These tests use pytest fixtures and parametrization for clean and efficient testing.
"""

import duckdb
import pytest

from src import config

pytestmark = pytest.mark.integration

# --- TEST CONFIGURATION ---

# 1. Define the tables we need to check
REQUIRED_TABLES = [
    "patients",
    "encounters",
    "conditions",
    "procedures",
    "medications",
]

# 2. Define the expected row counts for the full dataset.
#    This acts as a baseline to ensure the ETL processed everything.
#    It's good practice to keep these constants in one place.
EXPECTED_ROW_COUNTS = {
    "patients": 111278,
    "encounters": 5926090,
    "conditions": 3625440,
    "procedures": 15965984,
    "medications": 4451029,
}


# --- PYTEST FIXTURES ---


@pytest.fixture(scope="module")
def db_connection():
    """A pytest fixture that provides a read-only connection to the database."""
    if not config.DB_FILE.exists():
        pytest.fail(f"Database file not found at {config.DB_FILE}")

    con = duckdb.connect(database=str(config.DB_FILE), read_only=True)
    yield con
    con.close()


# --- TEST FUNCTIONS ---


def test_db_file_exists():
    """Test 1: Check if the final database file was created."""
    assert config.DB_FILE.exists(), f"Database file is missing: {config.DB_FILE}"


@pytest.mark.parametrize("table_name", REQUIRED_TABLES)
def test_table_is_not_empty(db_connection, table_name):
    """Test 2: A basic sanity check that each table has at least one row."""
    count = db_connection.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
    assert count > 0, f"Table '{table_name}' is empty."


@pytest.mark.full_dataset
@pytest.mark.parametrize(
    "table_name, expected_count", list(EXPECTED_ROW_COUNTS.items())
)
def test_table_has_expected_row_count(db_connection, table_name, expected_count):
    """
    Test 3: Verify each table has the exact number of rows from the full ETL.
    This is a crucial test to catch partially completed ETL runs.
    """
    count = db_connection.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
    print(f"Verifying row count for '{table_name}': Found {count:,} rows.")
    assert count == expected_count, (
        f"'{table_name}' has {count} rows, but {expected_count} were expected."
    )
