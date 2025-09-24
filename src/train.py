# src/train.py

import argparse
import logging
from pathlib import Path
import json

import catboost as cb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

# Import variables from our central configuration
from . import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def train_model(data_path: Path, model_path: Path):
    """
    Loads data, trains a weighted CatBoost classifier, finds the optimal
    decision threshold, and saves the model and threshold.
    """
    logging.info(f"Loading dataset from {data_path}...")
    df = pd.read_parquet(data_path)

    # --- 1. Define Features and Target ---
    TARGET = "readmitted_within_30_days"

    # Define all potential intermediate or identifier columns to remove
    features_to_drop = [
        "encounter_id", "patient_id", "admission_date", "discharge_date",
        "next_admission_date", "days_to_next_admission", "admission_reason_detail"
    ]
    
    # Filter list to only include columns that actually exist in the DataFrame
    cols_to_drop_existing = [col for col in features_to_drop if col in df.columns]

    X = df.drop(columns=[TARGET] + cols_to_drop_existing)
    y = df[TARGET]

    categorical_features = [
        "gender", "race", "marital_status", "admission_reason",
        "payer", "admission_day_of_week", "primary_diagnosis_code",
        "provider_id", "payer_dx_interaction",
    ]

    # --- 2. Split Data ---
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Save the test set for later evaluation ---
    logging.info("Saving the test set for external evaluation...")
    test_set_path = data_path.parent / "test_set.parquet"
    X_test.join(y_test).to_parquet(test_set_path)
    
    # --- 3. Preprocessing ---
    logging.info("Preprocessing categorical features for CatBoost...")
    for col in categorical_features:
        X_train[col] = X_train[col].astype(str).fillna("missing")
        X_test[col] = X_test[col].astype(str).fillna("missing")
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # --- 4. Train Weighted CatBoost Model ---
    logging.info("Calculating class weight for imbalance...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logging.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    model = cb.CatBoostClassifier(
        random_state=42,
        verbose=0,
        cat_features=config.CATEGORICAL_FEATURES,
        scale_pos_weight=scale_pos_weight,
    )
    
    logging.info("Training the weighted CatBoost model...")
    model.fit(X_train, y_train)

    # --- 5. Save the Model and the Pre-defined Threshold ---
    logging.info(f"Saving model to {model_path}...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    # THIS IS THE CHANGE:
    # We no longer calculate a threshold. We save the one from the config file.
    metadata_path = model_path.parent / "model_metadata.json"
    logging.info(f"Saving pre-defined threshold ({config.FINAL_THRESHOLD}) to {metadata_path}...")
    with open(metadata_path, 'w') as f:
        json.dump({"optimal_threshold": config.FINAL_THRESHOLD}, f)
        
    logging.info("✅ Model training and saving complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the readmission model.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("output/readmissions_dataset.parquet"),
        help="Path to the feature dataset (Parquet).",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("models/catboost_model.cbm"),
        help="Path to save the final model artifact.",
    )
    args = parser.parse_args()
    
    train_model(data_path=args.data_path, model_path=args.model_path)