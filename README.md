# Texas Hospital Readmission Prediction

This project demonstrates a complete, end-to-end MLOps workflow for predicting 30-day hospital readmissions. It begins with raw synthetic FHIR data and uses a containerized Python application to perform ETL, feature engineering, model training, and finally, deployment as a real-time REST API.

The project has been refactored from an initial exploratory notebook environment into a reproducible, script-based, and containerized application with automated CI/CD.

-----

## A Note on the Jupyter Notebooks

The `notebooks/` folder contains the original, step-by-step development of this project. They are the best resource for a detailed, **narrative-style walkthrough** of the exploratory data analysis (EDA), feature selection, and model tuning process. If you want to understand the **"why"** behind the final pipeline's design, the notebooks are the best place to start.

-----

## Project Status & Workflow

The project is organized into four main components: a batch pipeline for model training, a real-time API for serving predictions, a CI/CD pipeline for automation, and an interactive UI for user-friendly access.

### Part I: Batch Training Pipeline

The core logic is organized into a modular Python package (`src/`) and managed by an orchestration script (`pipeline.py`). This automated pipeline executes the following stages in sequence:

1.  **ETL (`src/etl.py`):** A parallelized pipeline parses raw FHIR JSON bundles, extracts relevant clinical and demographic data, and loads the clean, structured data into a **DuckDB** database.
2.  **Feature Engineering (`src/feature_engineering.py`):** SQL queries are executed against the DuckDB database to create the final analytical dataset. This step engineers the target variable (`readmitted_within_30_days`) and a rich feature set.
3.  **Model Training (`src/train.py`):** This script trains a weighted **CatBoost** classifier on the feature-engineered data. It saves the trained model and a metadata file containing the optimal decision threshold.
4.  **Evaluation (`src/evaluate.py`):** Using the saved artifacts, this final step generates a performance report and a confusion matrix on a hold-out test set to provide a clear assessment of the model.

### Part II: Real-Time Prediction API

A **FastAPI** application (`main.py`) serves the trained model through a REST endpoint.

1.  **Serving Layer (`main.py`):** This script loads the pre-trained CatBoost model and exposes a simple web endpoint.
2.  **Prediction Logic (`src/predict.py`):** This module contains the core logic to fetch data for a single patient encounter from the database, apply the *exact same* feature engineering steps used in training, and generate a real-time risk score.
3.  **Containerization (`Dockerfile`):** The entire application, including the API and all dependencies, is containerized with Docker, ensuring a consistent and reproducible environment for deployment.

### Part III: Continuous Integration (CI/CD)

This project is configured with a complete CI/CD pipeline using **GitHub Actions**. On every push to the `main` branch, the workflow automatically performs:

  * **Multi-Stage Testing:** Runs fast unit tests, executes the full training pipeline on a sample dataset for integration testing, and performs a smoke test on the live API server.
  * **Build & Publish:** If all tests pass, it builds the master Docker image.
  * **Versioning:** The image is tagged with the unique Git commit hash and pushed to **GitHub Container Registry (GHCR)**, ensuring a versioned, deployment-ready artifact is always available.

### Part IV: Interactive User Interface (UI)

A **Gradio** application (`app.py`) provides a user-friendly web interface for interacting with the trained model, making it accessible to non-technical stakeholders.

1.  **UI Layer (`app.py`):** This script loads the same trained model and metadata, creating a tabbed interface for predictions.
2.  **Two Prediction Modes:**
      * **Interactive Prediction:** A detailed form where users can input individual patient and encounter features to receive a real-time risk classification and probability.
      * **Predict from ID:** A dropdown menu pre-populated with known `encounter_id`s, allowing users to quickly see predictions for historical data.

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
docker run --rm -v ./data:/app/data -v ./output:/app/output -v ./models:/app/models readmission-api python pipeline.py
```

*Note: We use Docker volumes (`-v`) to ensure that the `output` and `models` generated inside the container are saved to your local machine.*

### Step 3: Run the Prediction API

Once the pipeline has successfully run and created the model, you can start the API server.

```bash
docker run --rm -p 8000:8000 -v ./output:/app/output -v ./models:/app/models readmission-api
```

The API will now be running and accessible.

### Step 4: Interact with the API

You can now get real-time predictions.

  * **Interactive Docs (Recommended):** Open your web browser and navigate to **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**. This interface allows you to test the endpoint directly.
  * **Command Line (`curl`):** Use a valid `encounter_id` from your dataset to get a prediction.
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

While the project is automated, you may need to perform ad-hoc data analysis or debug the code interactively. You can do this by launching a JupyterLab session inside the container. This gives you access to all the project's dependencies and output artifacts in a familiar notebook environment.

This command overrides the default command in the Dockerfile to start a JupyterLab server instead of the API.

**On Windows (Command Prompt):**

```bash
docker run -it --rm -p 8888:8888 -v "%cd%":/app readmission-api jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**On Linux, macOS, or Windows (PowerShell):**

```bash
docker run -it --rm -p 8888:8888 -v "$(pwd)":/app readmission-api jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

After running the command, copy the URL from your terminal (which includes a security token) and paste it into your web browser to begin your session.

-----

## Project Roadmap

### Completed

  * **User-Friendly UI:** A simple web interface has been developed using **Gradio**. This provides an interactive UI for making predictions, making the model accessible to non-technical users.

### Future Work

  * **Cloud Deployment:** Deploy the container from GHCR to a scalable cloud service like **AWS Elastic Container Service (ECS)** to create a production-grade, highly available prediction endpoint.
  * **Monitoring:** Implement logging and monitoring for the deployed API to track uptime, request latency, and potential model drift.

-----

## Data Generation

The raw data is generated using **Synthea™**, an open-source patient population simulator. To replicate the dataset, execute the following command from the root of the Synthea project directory.

```bash
java -jar synthea-with-dependencies.jar Texas -p 100000 -s 42 --exporter.fhir.use_us_core_ig true --exporter.csv.export true --exporter.fhir.export true
```

Place the generated `fhir` output folder into the `data/` directory of this project.

-----

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
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── .dockerignore
├── .gitignore
├── app.py
├── Dockerfile
├── main.py
├── pipeline.py
├── README.md
└── requirements.txt
```