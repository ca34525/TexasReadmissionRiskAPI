# Predicting 30-Day Hospital Readmissions

This document outlines the methodology and technical implementation for a project aimed at predicting 30-day hospital readmissions using synthetic FHIR data.

----------

## Introduction

Preventable hospital readmissions represent a significant challenge to the healthcare system, leading to increased costs and indicating potential gaps in patient care. This project presents a comprehensive data science approach to model and predict the likelihood of a patient being readmitted within 30 days of discharge. By leveraging synthetic patient data, we can develop and validate a model that could, in a real-world scenario, help clinicians identify at-risk patients and implement targeted interventions.

----------

## Project Setup

This project requires generating synthetic data and running the analysis environment within a Docker container.

### Part I: Data Generation

The raw data is generated using **Synthea™**, an open-source patient population simulator. To replicate the dataset, execute the following command from the root of the Synthea project directory. This will generate data for 100,000 patients in Texas and export it in both FHIR and CSV formats.

Bash

```
java -jar synthea-with-dependencies.jar Texas -p 100000 -s 42 --exporter.fhir.use_us_core_ig true --exporter.csv.export true --exporter.fhir.export true

```

Place the generated `fhir` and `csv` output folders into the `data/` directory of this project.

### Part II: Environment

The project environment is managed with Docker to ensure consistency.

1.  Build the Docker Image:
    
    This command builds the image using the provided Dockerfile, which includes all necessary dependencies like Python, DuckDB, and Jupyter Lab.
    
    Bash
    
    ```
    docker build -t readmissions-app .
    
    ```
    
2.  Run the Jupyter Lab Container:
    
    This command starts the Jupyter Lab server. It maps port 8888 for browser access and mounts the current project directory into the container's /app folder, allowing you to edit files locally.
    
    Bash
    
    ```
    docker run --rm -p 8888:8888 -v .:/app readmissions-app jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
    
    ```
    
    You can then access the Jupyter environment by navigating to `http://localhost:8888` in your browser and using the token provided in your terminal.
    

----------

## Project Workflow

The project is organized into a series of Jupyter notebooks, each with a distinct purpose. They are designed to be run in sequential order.

-   1_FHIR_ETL.ipynb
    
    This notebook handles the initial Extract, Transform, and Load (ETL) process. It parses the raw FHIR JSON bundles, flattens the necessary resources (Patient, Encounter, etc.), and loads them into a structured DuckDB database for efficient querying.
    
-   2_CSV_ETL.ipynb
    
    A validation notebook that loads the Synthea-generated CSV files into DuckDB. Its purpose is to compare the schemas and row counts against the FHIR-parsed data, ensuring the primary ETL pipeline is robust and accurate.
    
-   3_SQL_Feature_Engineering.ipynb
    
    This is where the core analytical dataset is constructed. Using SQL queries against the DuckDB database, this notebook engineers the primary features, including the crucial target variable (readmitted_within_30_days), length of stay, patient age, and historical admission counts.
    
-   4_EDA.ipynb
    
    A notebook dedicated to Exploratory Data Analysis. It examines the distributions of key variables, analyzes the relationship between various features and the readmission target, and provides a correlation matrix to inform the modeling stage.
    
-   5_Modeling.ipynb
    
    The final modeling notebook. It details the preprocessing steps, the training of two baseline models (Logistic Regression and XGBoost), and an evaluation of their performance using metrics like AUC-ROC and the classification report. The complete model pipelines are saved for future use.
    

----------

## Future Endeavors

The work completed thus far establishes a strong foundation. The next logical steps for this project involve operationalizing the model:

-   **API Deployment:** Develop a **FastAPI** endpoint to serve the trained XGBoost model, allowing for real-time prediction requests.
    
-   **User Interface:** Create a basic UI that can interact with the API. This will allow users to input patient characteristics and receive a readmission risk score, demonstrating the model's practical application.