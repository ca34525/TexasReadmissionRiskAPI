import json

import pandas as pd

from src import config
from src import train as train_module


def _training_frame(row_count: int = 20) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        payer = "Medicare" if index % 2 else "Aetna"
        diagnosis = "J18.9" if index % 2 else "I50.9"
        rows.append(
            {
                "encounter_id": f"enc-{index}",
                "length_of_stay": 2 + index,
                "age_at_admission": 40 + index,
                "gender": "female" if index % 2 else "male",
                "race": "white",
                "marital_status": "M",
                "admission_reason": "Inpatient admission",
                "payer": payer,
                "total_claim_cost": 1000.0 + index,
                "income": 50000 + index,
                "admission_day_of_week": "Monday",
                "primary_diagnosis_code": diagnosis,
                "provider_id": "provider-1",
                "payer_dx_interaction": f"{payer}_{diagnosis}",
                "prior_admissions_last_year": index % 3,
                "num_diagnoses": 1 + index % 2,
                "num_procedures": index % 2,
                "num_medications": index % 4,
                config.TARGET_VARIABLE: index % 2,
            }
        )
    return pd.DataFrame(rows)


def test_train_model_writes_compatible_artifacts(tmp_path, monkeypatch):
    data_path = tmp_path / "output" / "features.parquet"
    model_path = tmp_path / "models" / "model.cbm"
    data_path.parent.mkdir()
    _training_frame().to_parquet(data_path, index=False)

    created_models = []

    class FakeCatBoostClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_features = None
            created_models.append(self)

        def fit(self, features, target):
            self.fit_features = features.copy()
            self.fit_target = target.copy()

        def save_model(self, path):
            model_path = type(data_path)(path)
            model_path.write_bytes(b"test model")

    monkeypatch.setattr(
        train_module.cb,
        "CatBoostClassifier",
        FakeCatBoostClassifier,
    )

    train_module.train_model(data_path=data_path, model_path=model_path)

    assert model_path.read_bytes() == b"test model"
    assert json.loads(
        (model_path.parent / "model_metadata.json").read_text(encoding="utf-8")
    ) == {"optimal_threshold": config.FINAL_THRESHOLD}

    test_set = pd.read_parquet(data_path.parent / "test_set.parquet")
    assert config.TARGET_VARIABLE in test_set
    assert "encounter_id" not in test_set

    trained_model = created_models[0]
    assert trained_model.kwargs["cat_features"] == config.CATEGORICAL_FEATURES
    assert trained_model.kwargs["allow_writing_files"] is False
    assert trained_model.fit_features.columns.tolist() == [
        column
        for column in _training_frame().columns
        if column not in {"encounter_id", config.TARGET_VARIABLE}
    ]
    assert all(
        str(trained_model.fit_features[column].dtype) == "category"
        for column in config.CATEGORICAL_FEATURES
    )
