import pytest

from src import pipeline


def test_pipeline_runs_stages_in_order(monkeypatch, tmp_path):
    calls = []
    output_dir = tmp_path / "output"
    model_file = tmp_path / "models" / "model.cbm"

    monkeypatch.setattr(pipeline.config, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(pipeline.config, "DB_FILE", output_dir / "data.duckdb")
    monkeypatch.setattr(pipeline.config, "MODEL_FILE", model_file)
    monkeypatch.setattr(pipeline, "run_etl", lambda: calls.append("etl"))
    monkeypatch.setattr(
        pipeline,
        "create_features",
        lambda **kwargs: calls.append(("features", kwargs)),
    )
    monkeypatch.setattr(
        pipeline,
        "train_model",
        lambda **kwargs: calls.append(("train", kwargs)),
    )
    monkeypatch.setattr(pipeline, "evaluate_model", lambda: calls.append("evaluate"))

    pipeline.main()

    assert calls == [
        "etl",
        (
            "features",
            {
                "db_path": output_dir / "data.duckdb",
                "output_path": output_dir / "readmissions_dataset.parquet",
            },
        ),
        (
            "train",
            {
                "data_path": output_dir / "readmissions_dataset.parquet",
                "model_path": model_file,
            },
        ),
        "evaluate",
    ]


def test_pipeline_propagates_stage_failure(monkeypatch):
    def fail_etl():
        raise RuntimeError("invalid input")

    monkeypatch.setattr(pipeline, "run_etl", fail_etl)
    monkeypatch.setattr(
        pipeline,
        "create_features",
        lambda **kwargs: pytest.fail("feature engineering should not run"),
    )

    with pytest.raises(RuntimeError, match="invalid input"):
        pipeline.main()
