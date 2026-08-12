"""Loading and validation for the model artifact pair used at inference time."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import catboost as cb


class ModelArtifactsUnavailableError(RuntimeError):
    """Raised when the saved model and metadata cannot be used safely."""


@dataclass(frozen=True)
class ModelArtifacts:
    """A validated CatBoost model and its decision threshold."""

    model: cb.CatBoostClassifier
    threshold: float


def load_model_artifacts(model_path: Path) -> ModelArtifacts:
    """Load the matched model/metadata pair and validate its threshold."""
    try:
        model = cb.CatBoostClassifier()
        model.load_model(str(model_path))
    except (cb.CatBoostError, OSError) as exc:
        raise ModelArtifactsUnavailableError(
            "The model artifact could not be loaded."
        ) from exc

    metadata_path = model_path.parent / "model_metadata.json"
    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelArtifactsUnavailableError(
            "The model metadata could not be loaded."
        ) from exc

    threshold = (
        metadata.get("optimal_threshold") if isinstance(metadata, dict) else None
    )
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ModelArtifactsUnavailableError(
            "The model metadata contains an invalid decision threshold."
        )

    return ModelArtifacts(model=model, threshold=float(threshold))
