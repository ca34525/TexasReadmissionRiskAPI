import logging

# Import the main functions from each module in the 'src' package
from src.etl import main as run_etl
from src.feature_engineering import create_features
from src.train import train_model
from src.evaluate import evaluate_model

# Import the configuration variables that define our file paths
from src import config

# Configure logging
# ... (logging configuration remains the same) ...


def main():
    """
    Runs the full data processing and model training pipeline.
    """
    logging.info("="*50)
    logging.info("STARTING READMISSION PREDICTION PIPELINE")
    logging.info("="*50)

    # --- Define key file paths from the config module for clarity ---
    feature_dataset_path = config.OUTPUT_DIR / "readmissions_dataset.parquet"
    
    # --- Step 1: Run the ETL process ---
    logging.info("STEP 1: Starting ETL process...")
    try:
        run_etl()
        logging.info("✅ STEP 1: ETL process completed successfully.")
    except Exception as e:
        logging.error(f"💥 STEP 1: ETL process failed. Error: {e}")
        return # Stop the pipeline if ETL fails

    # --- Step 2: Run the feature engineering process ---
    logging.info("STEP 2: Starting feature engineering...")
    try:
        create_features(
            db_path=config.DB_FILE,
            output_path=feature_dataset_path
        )
        logging.info("✅ STEP 2: Feature engineering completed successfully.")
    except Exception as e:
        logging.error(f"💥 STEP 2: Feature engineering failed. Error: {e}")
        return # Stop the pipeline if this step fails

    # --- Step 3: Run the model training process ---
    logging.info("STEP 3: Starting model training...")
    try:
        # CHANGE: Use the MODEL_FILE variable from the config
        train_model(
            data_path=feature_dataset_path,
            model_path=config.MODEL_FILE
        )
        logging.info("✅ STEP 3: Model training completed successfully.")
    except Exception as e:
        logging.error(f"💥 STEP 3: Model training failed. Error: {e}")
        return # Stop the pipeline if training fails
        
    # --- Step 4: Run the final evaluation ---
    logging.info("STEP 4: Starting model evaluation...")
    try:
        evaluate_model()
        logging.info("✅ STEP 4: Evaluation completed successfully.")
    except Exception as e:
        logging.error(f"💥 STEP 4: Evaluation failed. Error: {e}")
        return # Stop if evaluation also fails

    # CHANGE: Move the final success message to the very end
    logging.info("="*50)
    logging.info("🎉 PIPELINE FINISHED SUCCESSFULLY 🎉")
    logging.info("="*50)


if __name__ == "__main__":
    main()