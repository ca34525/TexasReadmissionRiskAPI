# src/evaluate.py

import logging
from pathlib import Path
import json

import pandas as pd
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import variables from our central configuration
from . import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def evaluate_model():
    """
    Loads the test set, trained model, and optimal threshold to produce
    a detailed evaluation report and confusion matrix.
    """
    logging.info("Starting model evaluation...")
    
    # --- 1. Load Artifacts ---
    logging.info("Loading test set...")
    test_df = pd.read_parquet(config.OUTPUT_DIR / "test_set.parquet")
    
    logging.info("Loading trained model...")
    model = cb.CatBoostClassifier()
    model.load_model(config.MODEL_FILE)
    
    logging.info("Loading model metadata...")
    with open(config.MODELS_DIR / "model_metadata.json", 'r') as f:
        metadata = json.load(f)
    optimal_threshold = metadata["optimal_threshold"]
    
    # --- 2. Prepare Data ---
    X_test = test_df.drop(columns=[config.TARGET_VARIABLE])
    y_test = test_df[config.TARGET_VARIABLE]
    
    # --- THIS IS THE FIX ---
    # Apply the same preprocessing to the test set as was done during training.
    # This ensures the model receives data in the format it expects.
    logging.info("Preprocessing categorical features...")
    for col in config.CATEGORICAL_FEATURES:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str).fillna("missing")
            X_test[col] = X_test[col].astype("category")

    # --- 3. Make Predictions ---
    logging.info(f"Making predictions using threshold: {optimal_threshold:.4f}")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred_proba >= optimal_threshold).astype(int)
    
    # --- 4. Generate and Print Classification Report ---
    logging.info("--- Classification Report ---")
    report = classification_report(y_test, y_pred_class, target_names=["Not Readmitted", "Readmitted"])
    print(report)
    logging.info("\n" + report)
    
    # --- 5. Generate and Save Confusion Matrix ---
    logging.info("Generating and saving confusion matrix plot...")
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Predicted Not Readmitted", "Predicted Readmitted"],
                yticklabels=["Actual Not Readmitted", "Actual Readmitted"])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    confusion_matrix_path = config.OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    logging.info(f"Confusion matrix saved to {confusion_matrix_path}")

if __name__ == "__main__":
    evaluate_model()