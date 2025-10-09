# app.py

import json
import logging
from pathlib import Path

import catboost as cb
import gradio as gr
import pandas as pd

# --- Import project modules ---
from src import config
from src.predict import make_prediction

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- Load Model and Metadata at Startup ---
try:
    model = cb.CatBoostClassifier()
    model.load_model(str(config.MODEL_FILE))

    metadata_path = config.MODEL_FILE.parent / "model_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    THRESHOLD = metadata["optimal_threshold"]
    logging.info("Model and metadata loaded successfully for Gradio app.")
except Exception as e:
    logging.error(f"FATAL: Could not load model or metadata: {e}")
    model = None
    THRESHOLD = 0.5 # A default fallback

# =================================================================================
# TAB 1: INTERACTIVE PREDICTION LOGIC
# =================================================================================
def interactive_prediction(*features):
    if not model:
        return "Error: Model is not loaded. Please check server logs."
    
    try:
        # Create a dictionary from the input features
        feature_names = [
            'length_of_stay', 'age_at_admission', 'gender', 'race', 
            'marital_status', 'admission_reason', 'payer', 'total_claim_cost', 
            'income', 'admission_day_of_week', 'primary_diagnosis_code', 
            'provider_id', 'prior_admissions_last_year', 'num_diagnoses', 
            'num_procedures', 'num_medications'
        ]
        features_dict = dict(zip(feature_names, features))
        
        # Engineer the interaction feature
        features_dict["payer_dx_interaction"] = (
            str(features_dict.get("payer", "unknown")) + "_" +
            str(features_dict.get("primary_diagnosis_code", "unknown"))
        )
        
        # Create a single-row DataFrame and preprocess it
        df = pd.DataFrame([features_dict])
        for col in config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("missing").astype("category")
        df = df.reindex(columns=model.feature_names_, fill_value=0)
        
        # Make the prediction
        pred_proba = model.predict_proba(df)[0, 1]
        
        # Create the more descriptive classification text
        classification_text = "High Risk of Readmission" if pred_proba >= THRESHOLD else "Low Risk of Readmission"

        # Return a Markdown formatted string for bolding
        return (
            f"**Predicted Probability of Readmission:** {pred_proba:.2%}\n\n"
            f"**Classification:** {classification_text}"
        )

    except Exception as e:
        logging.error(f"Error during interactive prediction: {e}")
        return f"An error occurred: {e}"


# =================================================================================
# TAB 2: ID-BASED PREDICTION LOGIC
# =================================================================================
def id_based_prediction(encounter_id):
    if not encounter_id:
        return "Please enter an Encounter ID."
    
    try:
        result = make_prediction(encounter_id)
        if "error" in result:
            return f"Error: {result['error']}"

        probability = result['readmission_probability']
        classification_text = "High Risk of Readmission" if result['prediction'] == 1 else "Low Risk of Readmission"
        
        # Return a Markdown formatted string consistent with the other tab
        return (
            f"**Predicted Probability of Readmission:** {probability:.2%}\n\n"
            f"**Classification:** {classification_text}"
        )

    except Exception as e:
        logging.error(f"Error during ID-based prediction: {e}")
        return "Error: Could not retrieve prediction. Please check the Encounter ID."

# =================================================================================
# DEFINE THE GRADIO INTERFACES
# =================================================================================

# --- Interface for Tab 1 (Interactive Prediction) ---
interactive_inputs = [
    gr.Number(label="Length of Stay (days)", value=7),
    gr.Number(label="Age at Admission", value=50),
    gr.Radio(label="Gender", choices=["male", "female"], value="male"),
    gr.Dropdown(
        label="Race", 
        choices=["White", "Black or African American", "Asian", "American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander", "Unknown"], 
        value="White"
    ),
    gr.Dropdown(
        label="Marital Status", 
        choices=["M", "S", "D", "W"], 
        value="M"
    ),
    gr.Textbox(label="Admission Reason", value="Encounter for problem (procedure)"),
    gr.Dropdown(
        label="Payer", 
        choices=["Medicare", "NO_INSURANCE", "Cigna Health", "Aetna", "Anthem", "Humana", "Blue Cross Blue Shield", "Medicaid", "UnitedHealthcare", "Dual Eligible"], 
        value="Medicare"
    ),
    gr.Number(label="Total Claim Cost", value=26483),
    gr.Number(label="Income", value=74739),
    gr.Dropdown(
        label="Admission Day of Week", 
        choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
        value="Tuesday"
    ),
    gr.Textbox(label="Primary Diagnosis Code", value="424132000"),
    gr.Textbox(label="Provider ID", value="us-npi|9999868992"),
    gr.Number(label="Prior Admissions (Last Year)", value=2),
    gr.Number(label="Number of Diagnoses", value=1),
    gr.Number(label="Number of Procedures", value=9),
    gr.Number(label="Number of Medications", value=1)
]

interface1 = gr.Interface(
    fn=interactive_prediction,
    inputs=interactive_inputs,
    outputs=gr.Markdown(label="Prediction Result"),
    title="Interactive Prediction",
    description=(
        "Enter the patient's details below to predict their risk of readmission. Please use a valid SNOMED CT code for the primary diagnosis.<br><br>"
        "**Note:** The model's threshold is set to **70%** to effectively balance patient identification (74% recall and precision) "
        "with the need to reduce costly false alarms.<br>"
        "Therefore, patients with a less then 70% probability of readmission are labelled Low Risk"
    )
)

# --- Interface for Tab 2 (ID-Based Prediction) ---
# Updated input to be a Dropdown with the specified IDs
encounter_ids = [
    "ef5d7e9f-956d-2b7a-a4a6-c632f3b40cf9",
    "3c5e1be2-468a-e4d8-11f2-e767d59482d5",
    "6f06a6aa-a1da-bcd6-a43f-ddbbd638947c",
    "e2477992-082b-69ca-3152-6fecf4442626",
    "735f3287-d205-1ec8-9668-fcdac03f306a"
]

interface2 = gr.Interface(
    fn=id_based_prediction,
    inputs=gr.Dropdown(
        label="Encounter ID", 
        choices=encounter_ids, 
        value=encounter_ids[0]
    ),
    outputs=gr.Markdown(label="Prediction Result"), # Updated output to Markdown
    title="Predict from ID",
    description="Select a historical Encounter ID to retrieve its data and predict readmission risk."
)

# --- Combine interfaces into a single app with tabs ---
app = gr.TabbedInterface(
    [interface1, interface2],
    ["Interactive Prediction", "Predict from ID"]
)

# --- Launch the app ---
if __name__ == "__main__":
    app.launch()