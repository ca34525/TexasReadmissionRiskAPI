# AGENTS.md

## Scope and project map

- These instructions apply to the entire repository. Add a nested `AGENTS.md` or `AGENTS.override.md` only when a subtree develops materially different commands or rules.
- Run project commands from the repository root. `src/config.py` derives `PROJECT_ROOT`, `output/`, and `models/` from the current working directory.
- This is a Python 3.10 demonstration project for predicting 30-day hospital readmission from synthetic Synthea FHIR data.
- Production pipeline code lives in `src/`: ETL, feature engineering, CatBoost training, evaluation, and prediction.
- `main.py` is the FastAPI service, `app.py` is the Gradio UI, `tests/` contains automated checks, and `notebooks/` is exploratory/reference material rather than the production implementation.

## Setup and canonical commands

- Install development dependencies with `pip install -r requirements-dev.txt`;
  use `requirements.txt` for runtime-only environments.
- Run the default CI unit suite with `pytest`; check source with
  `ruff check .` and `ruff format --check .`.
- Run the complete pipeline from the repository root with `python -m src.pipeline`; set `FHIR_PATH` when the FHIR bundles are not under `data/fhir`.
- After a database has been generated, run the portable ETL integrity checks with `pytest -m "not full_dataset" tests/test_etl.py`.
- Run `pytest -m full_dataset tests/test_etl.py` only when the complete reference dataset is available; those tests assert exact full-dataset row counts.
- Use `docker build -t readmission-api .` when a change affects the container or deployment path.
- Ruff lint and format checks are configured in `pyproject.toml`. The repository does not currently configure a type checker or coverage gate; do not claim those checks passed or invent a required command.
- Prefer the actual module layout and `.github/workflows/ci-cd.yml` over stale command examples. The canonical pipeline entry point is `python -m src.pipeline`, not `python pipeline.py`.

## Implementation rules

- Make focused changes in the production modules; do not implement production fixes only in notebooks or duplicate source files under `tests/`.
- Keep training and serving features synchronized. A feature change may require coordinated edits to `src/feature_engineering.py`, `src/train.py`, `src/predict.py`, `main.py`, `app.py`, and `src/config.py`.
- Keep `config.CATEGORICAL_FEATURES`, categorical preprocessing, interaction-feature construction, and CatBoost feature ordering consistent across training, evaluation, ID-based prediction, and interactive prediction.
- Preserve the target definition: `readmitted_within_30_days` is based on the next inpatient admission after discharge. Do not introduce future information into model features.
- Treat `models/catboost_model.cbm` and `models/model_metadata.json` as a matched artifact pair. Regenerate or replace them only when model retraining is explicitly in scope.
- Preserve FastAPI request and response contracts. When a schema or endpoint changes, update the corresponding Gradio integration, tests, and user-facing documentation.
- Parameterize externally supplied SQL values. Any dynamic table or column names must come from a fixed allowlist, never directly from request data.
- Keep FastAPI and Gradio entry points distinct. Deployment changes must align the Docker command and exposed port, CI smoke test, and README instructions.
- Update tests when behavior changes. Do not weaken assertions simply to make a change pass.

## Data, artifacts, and operational safety

- Use only synthetic data in the repository. Never commit real patient data, protected health information, credentials, access tokens, or cloud secrets.
- `sample_data/` is a tracked synthetic fixture set. Do not replace it with real exports.
- Treat `data/` and `output/` as generated local state. Do not add their contents to version control.
- Avoid incidental changes to trained models, notebook outputs, `catboost_info/`, `notebooks/catboost_info/`, or `.gradio/flagged/` unless those artifacts are explicitly part of the task.
- Do not deploy, push images, access AWS resources, rotate credentials, or modify repository secrets unless the user explicitly requests that external action.
- Keep claims appropriately scoped: this repository is a demonstration model and must not be represented as validated clinical decision support without supporting evidence and approval.

## Validation by change type

- API or Pydantic changes: run `pytest tests/test_main.py`.
- Prediction, feature-alignment, or model-loading changes: run `pytest tests/test_predict.py` and the relevant API tests.
- ETL or DuckDB schema changes: run the pipeline on `sample_data/`, then `pytest -m "not full_dataset" tests/test_etl.py`.
- Feature engineering or training changes: run the relevant targeted tests plus the sample-data pipeline; verify the saved model and metadata remain compatible with both serving paths.
- Docker, entry-point, or port changes: build the image and verify the intended service's health/docs endpoint, then reconcile Dockerfile, CI, and README.
- Documentation-only changes: inspect the rendered Markdown and run `git diff --check`; application tests are optional unless commands or behavior were changed.
- If a required dependency, dataset, model artifact, Docker daemon, or cloud credential is unavailable, report the skipped check and the concrete reason. Never report an unrun check as passing.

## Code Review Rules

- Flag training-serving skew when feature names, transformations, categorical handling, or ordering differ. Safe path: update every affected training and serving path together and add a parity test.
- Flag target leakage or post-outcome data used as a predictor. Safe path: restrict features to information available at the index admission while using future admissions only to construct the target.
- Flag API contract drift between FastAPI, Gradio, tests, and documentation. Safe path: update all consumers in the same change or preserve backward compatibility.
- Flag committed PHI, non-synthetic patient data, credentials, or secrets. Safe path: remove the data from the change, use synthetic fixtures, and read secrets from the environment or repository secret store.
- Flag deployment changes where the container command, exposed port, CI smoke test, and README disagree. Safe path: choose the intended service and align all four surfaces.

## Completion expectations

- Keep the diff scoped and preserve unrelated user changes.
- Summarize behavior changes, tests run, tests skipped, and any generated artifacts intentionally updated.
- Update README or API documentation when public setup, execution, endpoint, or response behavior changes.
