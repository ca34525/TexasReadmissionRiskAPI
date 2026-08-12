"""Evaluate the saved model against the held-out test set."""

import logging

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .inference import prepare_model_features
from .model_artifacts import load_model_artifacts

logger = logging.getLogger(__name__)


def evaluate_model() -> None:
    """Log classification metrics and save a headless confusion-matrix plot."""
    logger.info("Starting model evaluation")
    test_df = pd.read_parquet(config.OUTPUT_DIR / "test_set.parquet")
    artifacts = load_model_artifacts(config.MODEL_FILE)

    features = test_df.drop(columns=[config.TARGET_VARIABLE])
    target = test_df[config.TARGET_VARIABLE]
    prepared = prepare_model_features(features, artifacts.model)

    logger.info("Making predictions using threshold %.4f", artifacts.threshold)
    probabilities = artifacts.model.predict_proba(prepared)[:, 1]
    predictions = (probabilities >= artifacts.threshold).astype(int)

    report = classification_report(
        target,
        predictions,
        target_names=["Not Readmitted", "Readmitted"],
        zero_division=0,
    )
    logger.info("Classification report:\n%s", report)

    matrix = confusion_matrix(target, predictions)
    figure = Figure(figsize=(8, 6))
    axis = figure.subplots()
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Not Readmitted", "Predicted Readmitted"],
        yticklabels=["Actual Not Readmitted", "Actual Readmitted"],
        ax=axis,
    )
    axis.set_title("Confusion Matrix")
    axis.set_ylabel("Actual Label")
    axis.set_xlabel("Predicted Label")
    figure.tight_layout()

    confusion_matrix_path = config.OUTPUT_DIR / "confusion_matrix.png"
    figure.savefig(confusion_matrix_path)
    logger.info("Confusion matrix saved to %s", confusion_matrix_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    evaluate_model()
