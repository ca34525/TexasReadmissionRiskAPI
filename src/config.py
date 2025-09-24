# src/config.py

"""
Central configuration file for the ETL pipeline and API.

This file defines key variables, such as file paths and processing parameters,
to ensure consistency and ease of modification across the project.
"""

import os
from pathlib import Path

# --- DIRECTORY SETUP ---
# NOTE: We use Path.cwd() to get the current working directory, which is
# typically the project's root folder where you run the scripts from.
PROJECT_ROOT = Path(os.getcwd())
DATA_DIR = PROJECT_ROOT / "data"
FHIR_DIR = DATA_DIR / "fhir"
OUTPUT_DIR = PROJECT_ROOT / "output"


# --- DATABASE CONFIGURATION ---
DB_FILE = OUTPUT_DIR / "synthea_fhir.duckdb"


# --- ETL PROCESSING PARAMETERS ---
# For a quick test, you can limit the number of files to process.
# Set to None to process all files.
MAX_FILES_TO_PROCESS = None

# This is the number of files to process in each chunk.
# Adjust this based on your system's RAM. 5000 is a safe start.
BATCH_SIZE = 5000

# Number of CPU cores to use for parallel processing.
# os.cpu_count() uses all available cores.
CPU_COUNT = os.cpu_count()