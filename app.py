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
# This mirrors the efficient loading in main.py
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
# This function mirrors the logic from your `/predict/interactive` FastAPI endpoint.
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
        prediction = "High Risk" if pred_proba >= THRESHOLD else "Low Risk"

        # Return a formatted dictionary for the Gradio Label component
        return {
            "High Risk": pred_proba if prediction == "High Risk" else 1 - pred_proba,
            "Low Risk": 1 - pred_proba if prediction == "High Risk" else pred_proba
        }
    except Exception as e:
        logging.error(f"Error during interactive prediction: {e}")
        return f"An error occurred: {e}"


# =================================================================================
# TAB 2: ID-BASED PREDICTION LOGIC
# This function is a wrapper around your existing 'make_prediction' function.
# =================================================================================
def id_based_prediction(encounter_id):
    if not encounter_id:
        return "Please enter an Encounter ID."
    
    try:
        result = make_prediction(encounter_id)
        if "error" in result:
            return f"Error: {result['error']}"

        prediction = "High Risk" if result['prediction'] == 1 else "Low Risk"
        probability = result['readmission_probability']
        
        return f"Prediction: {prediction}\nProbability: {probability:.2%}"

    except Exception as e:
        logging.error(f"Error during ID-based prediction: {e}")
        return "Error: Could not retrieve prediction. Please check the Encounter ID."

# =================================================================================
# DEFINE THE GRADIO INTERFACES
# =================================================================================

# --- Interface for Tab 1 (Interactive Prediction) ---
# The input components match the 'PredictionFeatures' Pydantic model, using examples from your main.py.
interactive_inputs = [
    gr.Number(label="Length of Stay (days)", value=7),
    gr.Number(label="Age at Admission", value=50),
    gr.Radio(label="Gender", choices=["male", "female"], value="male"),
    gr.Textbox(label="Race", value="White"),
    gr.Textbox(label="Marital Status", value="M"),
    gr.Textbox(label="Admission Reason", value="Encounter for problem (procedure)"),
    gr.Textbox(label="Payer", value="Medicare"),
    gr.Number(label="Total Claim Cost", value=26483),
    gr.Number(label="Income", value=74739),
    gr.Textbox(label="Admission Day of Week", value="Tuesday"),
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
    outputs=gr.Label(label="Prediction Result"),
    title="Interactive Prediction",
    description="Fill in the patient and encounter details below to get a real-time readmission risk prediction."
)

# --- Interface for Tab 2 (ID-Based Prediction) ---
interface2 = gr.Interface(
    fn=id_based_prediction,
    inputs=gr.Textbox(label="Encounter ID", placeholder="Enter a valid encounter_id from the database..."),
    outputs=gr.Textbox(label="Prediction Result", lines=2),
    title="Predict from ID",
    description="Enter a historical Encounter ID to retrieve its data and predict readmission risk."
)

# --- Combine interfaces into a single app with tabs ---
app = gr.TabbedInterface(
    [interface1, interface2],
    ["Interactive Prediction", "Predict from ID"]
)

# --- Launch the app ---
if __name__ == "__main__":
    app.launch()