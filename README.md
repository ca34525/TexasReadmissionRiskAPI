# Texas Hospital Readmission Prediction

This project demonstrates a complete, end-to-end workflow for predicting 30-day hospital readmissions. It begins with raw synthetic FHIR data and uses a containerized Python application to perform ETL, feature engineering, model training, and finally, deployment as a real-time REST API.

The project has been refactored from an initial exploratory notebook environment into a reproducible, script-based, and containerized application with automated CI/CD.

*This project uses synthetic data and is an engineering demonstration, not validated clinical decision-support software.*

-----

## Where to Start

There are two great ways to get familiar with this project, depending on your goal.

* **For the quickest path to see the model in action**, use the live, interactive demo deployed on AWS App Runner. This UI lets you get real-time predictions without running any code.
    **Access the live demo here:** **[https://pvv8v4igm6.us-east-1.awsapprunner.com/](https://pvv8v4igm6.us-east-1.awsapprunner.com/)**

* **To understand the "why" behind the project**, the Jupyter notebooks are the best resource. They provide a detailed, narrative-style walkthrough of the exploratory data analysis (EDA), feature selection, and model tuning process.

-----

## Data Generation

The raw data is generated using **Synthea™**, an open-source patient population simulator. To replicate the dataset, execute the following command from the root of the Synthea project directory.

```bash
java -jar synthea-with-dependencies.jar Texas -p 100000 -s 42 --exporter.fhir.use_us_core_ig true --exporter.csv.export true --exporter.fhir.export true
```

Place the generated `fhir` output folder into the `data/` directory of this project.

## Project Status & Workflow

The project is organized into four main components: a batch pipeline for model training, a real-time API for serving predictions, a CI/CD pipeline for automation, and an interactive UI for user-friendly access.

### Part I: Batch Training Pipeline

The core logic is organized into a modular Python package (`src/`) and managed by an orchestration script (`src/pipeline.py`). This automated pipeline executes the following stages in sequence:

1.  **ETL (`src/etl.py`):** A parallelized pipeline parses raw FHIR JSON bundles, extracts relevant clinical and demographic data, and loads the clean, structured data into a **DuckDB** database.
2.  **Feature Engineering (`src/feature_engineering.py`):** SQL queries are executed against the DuckDB database to create the final analytical dataset. This step engineers the target variable (`readmitted_within_30_days`) and a rich feature set.
3.  **Model Training (`src/train.py`):** This script trains a weighted **CatBoost** classifier on the feature-engineered data. It saves the trained model and a metadata file containing the optimal decision threshold.
4.  **Evaluation (`src/evaluate.py`):** Using the saved artifacts, this final step generates a performance report and a confusion matrix on a hold-out test set to provide a clear assessment of the model.

### Part II: Real-Time Prediction API

A **FastAPI** application (`main.py`) serves the trained model through a REST endpoint.

1.  **Serving Layer (`main.py`):** This script loads the pre-trained CatBoost model and exposes a simple web endpoint.
2.  **Prediction Logic (`src/predict.py`):** This module contains the core logic to fetch data for a single patient encounter from the database, apply the *exact same* feature engineering steps used in training, and generate a real-time risk score.
3.  **Containerization (`Dockerfile`):** The entire application, including the API and all dependencies, is containerized with Docker, ensuring a consistent and reproducible environment for deployment.

### Part III: Continuous Integration

This project uses **GitHub Actions** for credential-free checks on pull requests and pushes to `main`. The workflow:

* Runs the unit tests and Ruff source checks.
* Builds the Docker image and runs the full pipeline on the tracked synthetic sample data.
* Runs portable ETL integrity tests and a FastAPI readiness smoke test.

Image publishing is kept in a separate, manually triggered workflow so normal CI does not require AWS credentials or modify cloud resources.

### Part IV: Interactive User Interface (UI)

A **Gradio** application (`app.py`) provides a user-friendly web interface for interacting with the trained model, making it accessible to non-technical stakeholders.

1.  **UI Layer (`app.py`):** This script loads the same trained model and metadata, creating a tabbed interface for predictions.
2.  **Two Prediction Modes:**
      * **Interactive Prediction:** A detailed form where users can input individual patient and encounter features to receive a real-time risk classification and probability.
      * **Predict from ID:** A dropdown populated from the generated DuckDB database for predictions on historical synthetic encounters.

-----

## How to Use This Project

The entire project is managed with Docker for complete reproducibility.

### Prerequisites

  * Docker installed and running on your machine.
  * Synthea™-generated data placed in the `./data/` directory (see "Data Generation" section).

### Step 1: Build the Master Docker Image

First, build the Docker image that contains all dependencies for both the pipeline and the API. This command only needs to be run once.

```bash
docker build -t readmission-api .
```

### Step 2: Run the Training Pipeline

Next, run the end-to-end training pipeline inside the container. This will execute the ETL, feature engineering, training, and evaluation steps. The process will create the DuckDB database in `output/` and the trained model in `models/`, which are required for the API and UI to function.

```bash
docker run --rm -v ./data:/app/data -v ./output:/app/output -v ./models:/app/models readmission-api python -m src.pipeline
```

*Note: We use Docker volumes (`-v`) to ensure that the `output` and `models` generated inside the container are saved to your local machine.*

### Step 3: Run the Prediction API

Once the pipeline has successfully run and created the model, you can start the API server by overriding the default Gradio command with Uvicorn.

```bash
docker run --rm -p 8000:8000 -v ./output:/app/output -v ./models:/app/models readmission-api python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 4: Interact with the API

You can now get real-time predictions.

  * **Interactive Docs (Recommended):** Open **[http://localhost:8000/docs](http://localhost:8000/docs)**.
  * **Command Line (`curl`):** Use a valid `encounter_id` from your dataset.

```bash
curl http://127.0.0.1:8000/predict/your-encounter-id-here
```

### Step 5: Launch the Interactive UI (Alternative)

As an alternative to the raw API, you can launch the Gradio web interface to interact with the model.

```bash
docker run --rm -p 7860:7860 -v ./output:/app/output -v ./models:/app/models readmission-api python app.py
```

Open your web browser and navigate to **[http://127.0.0.1:7860](http://127.0.0.1:7860)** to use the application.

### Running an Interactive Analysis Session (Optional)

Notebook dependencies are kept separate from the runtime image. For local notebook work, install them and launch JupyterLab from the repository root:

```bash
python -m pip install -r requirements-notebooks.txt
jupyter lab
```

### Local Checks

Install development dependencies and run the same fast checks used by CI:

```bash
python -m pip install -r requirements-dev.txt
pytest
ruff check .
ruff format --check .
```

-----

## Cloud Deployment on AWS App Runner

This project's interactive UI has been successfully deployed to **AWS App Runner**, a fully managed service for containerized applications. This provides a scalable, managed public endpoint.

The deployment workflow leverages a container-native approach. The final Docker image, which bundles the application and all its dependencies, is first pushed to a private repository in **Amazon Elastic Container Registry (ECR)**.

An AWS App Runner service is then configured to pull this image directly from the ECR repository. This setup utilizes an **IAM role** to securely grant App Runner the necessary permissions for access. The service is set for automatic deployments, meaning any new image pushed to ECR will trigger an update to the live application. Finally, the service is configured to expose port **`7860`** for the Gradio UI.

## Project Structure

```
ReadmissionRiskAPI/
├── data/
├── models/
├── notebooks/
├── output/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── etl.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── inference.py
│   ├── model_artifacts.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── .dockerignore
├── .gitignore
├── app.py
├── Dockerfile
├── main.py
├── README.md
├── requirements-dev.txt
├── requirements-notebooks.txt
└── requirements.txt
```
