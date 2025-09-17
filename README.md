Synthea data: java -jar synthea-with-dependencies.jar Texas -p 100000 -s 42 --exporter.fhir.use_us_core_ig true --exporter.csv.export true --exporter.fhir.export true

Build Image: docker build -t readmissions-app .

Launch the Jupyter Notebook Server: docker run --rm -p 8888:8888 -v .:/app readmissions-app jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

Access Notebook: file:///root/.local/share/jupyter/runtime/jpserver-1-open.html


# Hospital Readmission Risk Prediction API

## Objective

This project aims to predict the risk of hospital readmission for patients using synthetic health data. The process involves ingesting and processing 100,000 FHIR bundles, engineering features, training a machine learning model, and deploying the model as a web service with a user interface.

## Project Workflow

1.  **ETL (Extract, Transform, Load)**

      * **FHIR to DuckDB:** Raw FHIR JSON bundles from the `/data/fhir` directory are processed in parallel. Key resources (Patient, Encounter, Condition, Procedure, MedicationRequest) are parsed, cleaned, and loaded into a centralized DuckDB database (`/output/synthea_fhir.duckdb`).
      * **Data Validation:** The data loaded from FHIR bundles is compared against the original source CSVs (`/data/csv`) to verify row counts and schema integrity.

2.  **Exploratory Data Analysis (EDA) & Feature Engineering**

      * **Data Exploration:** Analyze the distributions, relationships, and quality of the data in DuckDB to inform feature selection.
      * **Define Target Variable:** Create the binary outcome label for the model. A common definition is **"a patient is readmitted if they have an unplanned inpatient encounter within 30 days of being discharged from a previous inpatient encounter."**
      * **Feature Creation:** Engineer features from the patient's history prior to each discharge. This includes demographic data, encounter history (frequency, length of stay), diagnoses (e.g., Charlson Comorbidity Index), procedures, and medication history.

3.  **Model Training and Evaluation**

      * **Data Preprocessing:** Prepare the feature set for modeling by handling missing values, encoding categorical variables, and scaling numerical features.
      * **Model Training:** Split the data into training and testing sets. Train a classification model (e.g., Logistic Regression, LightGBM, XGBoost) to predict the readmission target variable.
      * **Evaluation:** Assess model performance using appropriate metrics like AUC-ROC, Precision-Recall, and F1-Score.
      * **Serialization:** Save the final trained model (e.g., as a `.joblib` or `.pkl` file) for deployment.

4.  **Deployment**

      * **API Development:** Create a REST API using **FastAPI** that exposes a `/predict` endpoint. This endpoint will accept patient data in a defined format, load the serialized model, and return a readmission risk prediction.
      * **Containerization:** Package the FastAPI application, the trained model, and all dependencies into a **Docker** image for consistent and portable deployment.
      * **User Interface (UI):** Develop a simple web interface using a framework like **Streamlit** or **Gradio**. This UI will interact with the FastAPI backend, allowing users to input patient information and view the model's prediction.

## Directory Structure

```
.
├── data/               # Raw input data (FHIR, CSV)
├── notebooks/          # Jupyter notebooks for ETL, EDA, and modeling
├── output/             # Generated files, including the DuckDB database
├── src/                # Source code for the FastAPI app and modeling pipeline
├── .gitignore          # Specifies files for Git to ignore
├── Dockerfile          # Instructions for building the Docker image
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd ReadmissionRiskAPI
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the ETL Pipeline:**
    Execute the notebooks in order to populate the DuckDB database.

      * `notebooks/FHIR ETL.ipynb`
      * `notebooks/CSV ETL.ipynb` (for validation)

2.  **Train the Model:**
    Run the model training notebook or script.

      * `notebooks/Model Training.ipynb`

3.  **Run the API:**
    Launch the FastAPI application locally.

    ```bash
    uvicorn src.main:app --reload
    ```

4.  **Build and Run with Docker:**

    ```bash
    # Build the Docker image
    docker build -t readmission-risk-api .

    # Run the Docker container
    docker run -p 8000:8000 readmission-risk-api
    ```

## Technology Stack

  * **Data Processing:** Python, DuckDB, Pandas
  * **Modeling:** Scikit-learn, XGBoost/LightGBM
  * **API:** FastAPI
  * **Deployment:** Docker, Uvicorn
  * **UI (Proposed):** Streamlit or Gradio