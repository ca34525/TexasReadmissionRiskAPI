"""Orchestrate the end-to-end readmission modeling pipeline."""

import logging
from collections.abc import Callable

from src import config
from src.etl import main as run_etl
from src.evaluate import evaluate_model
from src.feature_engineering import create_features
from src.train import train_model

logger = logging.getLogger(__name__)


def _run_stage(name: str, operation: Callable[[], None]) -> None:
    """Run one pipeline stage and preserve a failing process exit status."""
    logger.info("Starting %s", name)
    try:
        operation()
    except Exception:
        logger.exception("Pipeline stage failed: %s", name)
        raise
    logger.info("Completed %s", name)


def main() -> None:
    """Run ETL, feature engineering, training, and evaluation in order."""
    feature_dataset_path = config.OUTPUT_DIR / "readmissions_dataset.parquet"

    logger.info("Starting readmission prediction pipeline")
    _run_stage("ETL", run_etl)
    _run_stage(
        "feature engineering",
        lambda: create_features(
            db_path=config.DB_FILE,
            output_path=feature_dataset_path,
        ),
    )
    _run_stage(
        "model training",
        lambda: train_model(
            data_path=feature_dataset_path,
            model_path=config.MODEL_FILE,
        ),
    )
    _run_stage("model evaluation", evaluate_model)
    logger.info("Readmission prediction pipeline finished successfully")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
