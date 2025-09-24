import argparse
import logging
from pathlib import Path

import catboost as cb
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def train_model(data_path: Path, model_path: Path):
    """
    Loads the feature-engineered dataset, trains a weighted CatBoost
    classifier, and saves the final model artifact.

    Args:
        data_path (Path): Path to the feature dataset (Parquet).
        model_path (Path): Path to save the trained model file (.cbm).
    """
    logging.info(f"Loading dataset from {data_path}...")
    df = pd.read_parquet(data_path)

    # --- 1. Define Features and Target ---
    TARGET = "readmitted_within_30_days"
    
    # --- THIS IS THE FIX ---
    # Add 'admission_date' to the list of columns to drop.
    features_to_drop = [
        "encounter_id", "patient_id", "admission_reason_detail", "admission_date"
    ]
    
    X = df.drop(columns=[TARGET] + features_to_drop)
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

    # --- 3. Preprocessing for CatBoost ---
    # CatBoost requires categorical features with NaNs to be handled.
    logging.info("Preprocessing categorical features for CatBoost...")
    for col in categorical_features:
        X_train[col] = X_train[col].astype(str).fillna("missing")
        X_test[col] = X_test[col].astype(str).fillna("missing")
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # --- 4. Train Final Weighted CatBoost Model ---
    logging.info("Calculating class weight for imbalance...")
    # Weight = count of majority class / count of minority class
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logging.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    model = cb.CatBoostClassifier(
        random_state=42,
        verbose=0,
        cat_features=categorical_features,
        scale_pos_weight=scale_pos_weight,
    )

    logging.info("Training the final CatBoost model...")
    model.fit(X_train, y_train)

    # --- 5. Save the Model ---
    logging.info(f"Saving model to {model_path}...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

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