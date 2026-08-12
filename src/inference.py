"""Shared feature preparation and scoring for every serving entry point."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from . import config


def prepare_model_features(features: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Align a feature frame to the trained CatBoost model's schema."""
    prepared = features.reindex(columns=model.feature_names_, fill_value=0).copy()
    for column in config.CATEGORICAL_FEATURES:
        if column in prepared.columns:
            prepared[column] = (
                prepared[column].astype(str).fillna("missing").astype("category")
            )
    return prepared


def predict_feature_frame(
    model: Any,
    threshold: float,
    features: pd.DataFrame,
) -> dict[str, float | int]:
    """Score an already engineered feature frame."""
    prepared = prepare_model_features(features, model)
    probability = float(model.predict_proba(prepared)[0, 1])
    return {
        "readmission_probability": probability,
        "prediction": int(probability >= threshold),
        "threshold": float(threshold),
    }


def predict_feature_mapping(
    model: Any,
    threshold: float,
    features: Mapping[str, Any],
) -> dict[str, float | int]:
    """Engineer the interactive-only feature and score one input mapping."""
    feature_values = dict(features)
    feature_values["payer_dx_interaction"] = (
        f"{feature_values.get('payer', 'unknown')}_"
        f"{feature_values.get('primary_diagnosis_code', 'unknown')}"
    )
    return predict_feature_frame(
        model,
        threshold,
        pd.DataFrame([feature_values]),
    )
