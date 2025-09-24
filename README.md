# Texas Hospital Readmission Prediction

This project demonstrates an end-to-end, automated workflow for predicting 30-day hospital readmissions. It begins with raw synthetic FHIR data and uses a containerized Python pipeline to perform ETL, feature engineering, model training, and evaluation.

The project has been refactored from an initial exploratory notebook environment into a reproducible, script-based application, ready for deployment.

### A Note on the Jupyter Notebooks

While the final, automated workflow is managed by the Python scripts in the `src/` directory, the `notebooks/` folder contains the original, step-by-step development of this project.

The notebooks are the best resource for a detailed, **narrative-style walkthrough**. They contain extensive commentary, exploratory data analysis (EDA) visualizations, and the incremental process of model development and tuning. If you want to understand the **"why"** behind the final pipeline's design, the notebooks are the best place to start. 

#### How to Run the Notebooks

To explore the project in the original Jupyter environment, first, build the Docker image (as described in "Part II") and then run the Jupyter Lab server with this command:

```bash
docker run --rm -p 8888:8888 -v .:/app readmissions-app jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

You can then access the environment by opening the `http://localhost:8888` link in your browser and using the token provided in your terminal.

## Project Workflow

The core logic is organized into a modular Python package (`src/`) and managed by a single orchestration script (`pipeline.py`). This automated pipeline executes the following stages in sequence:

1.  **ETL (`src/etl.py`):** A parallelized pipeline parses raw FHIR JSON bundles, extracts relevant clinical and demographic data, and loads the clean, structured data into a **DuckDB** database.
2.  **Feature Engineering (`src/feature_engineering.py`):** SQL queries are executed against the DuckDB database to create the final analytical dataset. This step engineers the target variable (`readmitted_within_30_days`) and a rich feature set.
3.  **Model Training (`src/train.py`):** This script trains a weighted **CatBoost** classifier on the feature-engineered data. It saves three key artifacts: the trained model, the test set for evaluation, and a metadata file containing the optimal decision threshold.
4.  **Evaluation (`src/evaluate.py`):** Using the saved artifacts, this final step generates a performance report and a confusion matrix to provide a clear assessment of the model on the hold-out test data.

## Next Steps: API Deployment

With the core data processing and training pipeline complete, the next phase is to deploy the model as a real-time prediction service.

  - **[COMPLETED]** \~\~Refactor Code: Convert logic from Jupyter notebooks into modular Python scripts.\~\~
  - **[NEXT] Prediction API:** Build a REST API using **FastAPI**. The API will load the trained CatBoost model and provide an endpoint that accepts an `encounter_id` and returns a real-time readmission risk score.
  - **User Interface:** Develop a simple front-end to interact with the API.
  - **Containerization:** Containerize the final FastAPI application for seamless deployment.

## Part I: Data Generation

The raw data is generated using **Synthea™**, an open-source patient population simulator. To replicate the dataset, execute the following command from the root of the Synthea project directory. This will generate data for 100,000 patients in Texas and export it in both FHIR and CSV formats.

```bash
java -jar synthea-with-dependencies.jar Texas -p 100000 -s 42 --exporter.fhir.use_us_core_ig true --exporter.csv.export true --exporter.fhir.export true
```

Place the generated `fhir` and `csv` output folders into the `data/` directory of this project.

## Part II: Environment Setup and Usage

The project is managed with Docker for complete reproducibility.

#### 1\. Build the Docker Image

This command builds the image using the provided Dockerfile, which installs all Python dependencies.

```bash
docker build -t readmissions-app .
```

#### 2\. Run the Full Pipeline

This command runs the entire end-to-end pipeline inside the container. It will execute the ETL, feature engineering, training, and evaluation steps, saving all outputs to the `output/` and `models/` directories.

```bash
docker run --rm -v .:/app readmissions-app python pipeline.py
```

Upon completion, you can review the performance report in the console and find the generated confusion matrix plot in `output/confusion_matrix.png`.

#### (Optional) Run an Interactive Shell

To explore the container's environment or run scripts manually, use this command:

```bash
docker run --rm -it -v .:/app readmissions-app /bin/bash
```

## Project Structure

```
ReadmissionRiskAPI/
├── data/                 # Raw Synthea data (FHIR and CSV)
├── models/               # Saved model artifacts (.cbm and .json)
├── notebooks/            # Original notebooks for exploration and reference
├── output/               # Intermediate and final data outputs (.duckdb, .parquet)
├── src/                  # Source code for the ETL and ML pipeline
│   ├── __init__.py
│   ├── config.py         # Central configuration file
│   ├── etl.py            # FHIR to DuckDB ETL script
│   ├── evaluate.py       # Model evaluation script
│   ├── feature_engineering.py # Feature engineering script
│   ├── train.py          # Model training script
│   └── utils.py          # Helper functions for ETL
├── .gitignore
├── Dockerfile            # Defines the container environment
├── pipeline.py           # Master orchestration script
├── README.md             # This file
└── requirements.txt      # Python dependencies
```