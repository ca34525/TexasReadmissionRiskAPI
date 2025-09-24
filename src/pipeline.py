# pipeline.py

import logging

# Import the main functions from each module in the 'src' package
from src.etl import main as run_etl
from src.feature_engineering import create_features
from src.train import train_model
from src.evaluate import evaluate_model

# Import the configuration variables that define our file paths
from src import config

# Configure logging to monitor the pipeline's progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)


def main():
    """
    Runs the full data processing and model training pipeline.
    """
    logging.info("="*50)
    logging.info("STARTING READMISSION PREDICTION PIPELINE")
    logging.info("="*50)

    # --- Step 1: Run the ETL process ---
    # This processes raw FHIR JSON files into a structured DuckDB database.
    logging.info("STEP 1: Starting ETL process...")
    try:
        run_etl()
        logging.info("✅ STEP 1: ETL process completed successfully.")
    except Exception as e:
        logging.error(f"💥 STEP 1: ETL process failed. Error: {e}")
        return # Stop the pipeline if ETL fails

    # --- Step 2: Run the feature engineering process ---
    # This uses the DuckDB database to create the final analytical dataset.
    logging.info("STEP 2: Starting feature engineering...")
    try:
        # Define the path for the engineered features dataset.
        # This is the output of this step and the input for the next.
        feature_dataset_path = config.OUTPUT_DIR / "readmissions_dataset.parquet"
        
        create_features(
            db_path=config.DB_FILE,
            output_path=feature_dataset_path
        )
        logging.info("✅ STEP 2: Feature engineering completed successfully.")
    except Exception as e:
        logging.error(f"💥 STEP 2: Feature engineering failed. Error: {e}")
        return # Stop the pipeline if this step fails

    # --- Step 3: Run the model training process ---
    # This trains the CatBoost classifier and saves the final model artifact.
    logging.info("STEP 3: Starting model training...")
    try:
        # The model path is defined in the config.
        # We use the feature dataset created in the previous step.
        train_model(
            data_path=feature_dataset_path,
            model_path=config.MODELS_DIR / "catboost_model.cbm"
        )
        logging.info("✅ STEP 3: Model training completed successfully.")
    except Exception as e:
        logging.error(f"💥 STEP 3: Model training failed. Error: {e}")
        return # Stop the pipeline if training fails
        
    logging.info("="*50)
    logging.info("🎉 PIPELINE FINISHED SUCCESSFULLY 🎉")
    logging.info("="*50)

    # --- Step 4: Run the final evaluation ---
    logging.info("STEP 4: Starting model evaluation...")
    try:
        evaluate_model()
        logging.info("✅ STEP 4: Evaluation completed successfully.")
    except Exception as e:
        logging.error(f"💥 STEP 4: Evaluation failed. Error: {e}")


if __name__ == "__main__":
    main()