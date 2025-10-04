# src/config.py

"""
Central configuration file for the ETL pipeline and API.

This file defines key variables, such as file paths and processing parameters,
to ensure consistency and ease of modification across the project.
"""

import os
from pathlib import Path

# --- DIRECTORY SETUP ---
# Use an environment variable for the FHIR data path,
# falling back to the local project structure if not set.
FHIR_DIR = Path(os.getenv("FHIR_PATH", Path(os.getcwd()) / "data" / "fhir"))

# The rest of the paths remain the same
PROJECT_ROOT = Path(os.getcwd())
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"


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

# --- MODELING & API CONFIGURATION ---
MODEL_FILE = MODELS_DIR / "catboost_model.cbm"
TARGET_VARIABLE = "readmitted_within_30_days"

# List of categorical features for the model
CATEGORICAL_FEATURES = [
    "gender", "race", "marital_status", "admission_reason",
    "payer", "admission_day_of_week", "primary_diagnosis_code",
    "provider_id", "payer_dx_interaction",
]

# The decision threshold determined from the notebook analysis.
FINAL_THRESHOLD = 0.7