# app.py

import logging

import gradio as gr

# --- Import project modules ---
from src import config
from src.inference import predict_feature_mapping
from src.model_artifacts import (
    ModelArtifactsUnavailableError,
    load_model_artifacts,
)
from src.predict import list_inpatient_encounter_ids, make_prediction

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- Load Model and Metadata at Startup ---
try:
    artifacts = load_model_artifacts(config.MODEL_FILE)
    model = artifacts.model
    THRESHOLD = artifacts.threshold
    logging.info("Model and metadata loaded successfully for Gradio app.")
except ModelArtifactsUnavailableError:
    logging.exception("Could not load model or metadata for the Gradio app.")
    model = None
    THRESHOLD = float(config.FINAL_THRESHOLD)


# =================================================================================
# TAB 1: INTERACTIVE PREDICTION LOGIC
# =================================================================================
def interactive_prediction(*features):
    if model is None:
        return "Error: Model is not loaded. Please check server logs."

    try:
        # Create a dictionary from the input features
        feature_names = [
            "length_of_stay",
            "age_at_admission",
            "gender",
            "race",
            "marital_status",
            "admission_reason",
            "payer",
            "total_claim_cost",
            "income",
            "admission_day_of_week",
            "primary_diagnosis_code",
            "provider_id",
            "prior_admissions_last_year",
            "num_diagnoses",
            "num_procedures",
            "num_medications",
        ]
        features_dict = dict(zip(feature_names, features))

        result = predict_feature_mapping(model, THRESHOLD, features_dict)
        pred_proba = result["readmission_probability"]

        # Create the more descriptive classification text
        classification_text = (
            "High Risk of Readmission"
            if pred_proba >= THRESHOLD
            else "Low Risk of Readmission"
        )

        # Return a Markdown formatted string for bolding
        return (
            f"**Predicted Probability of Readmission:** {pred_proba:.2%}\n\n"
            f"**Classification:** {classification_text}"
        )

    except Exception:
        logging.exception("Error during interactive prediction.")
        return "An error occurred while generating the prediction."


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

        probability = result["readmission_probability"]
        classification_text = (
            "High Risk of Readmission"
            if result["prediction"] == 1
            else "Low Risk of Readmission"
        )

        # Return a Markdown formatted string consistent with the other tab
        return (
            f"**Predicted Probability of Readmission:** {probability:.2%}\n\n"
            f"**Classification:** {classification_text}"
        )

    except Exception:
        logging.exception("Error during ID-based prediction.")
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
        choices=[
            "White",
            "Black or African American",
            "Asian",
            "American Indian or Alaska Native",
            "Native Hawaiian or Other Pacific Islander",
            "Unknown",
        ],
        value="White",
    ),
    gr.Dropdown(label="Marital Status", choices=["M", "S", "D", "W"], value="M"),
    gr.Dropdown(
        label="Admission Reason",
        choices=[
            "Encounter for problem (procedure)",
            "Hospital admission (procedure)",
            "Drug rehabilitation and detoxification (regime/therapy)",
            "Admission to surgical department (procedure)",
            "Admission to intensive care unit (procedure)",
            "Admission to ward (procedure)",
            "Patient transfer to intensive care unit (procedure)",
            "Admission to surgical transplant department (procedure)",
            "Non-urgent orthopedic admission (procedure)",
            "Hospital admission for isolation (procedure)",
        ],
        value="Encounter for problem (procedure)",
    ),
    gr.Dropdown(
        label="Payer",
        choices=[
            "Medicare",
            "NO_INSURANCE",
            "Cigna Health",
            "Aetna",
            "Anthem",
            "Humana",
            "Blue Cross Blue Shield",
            "Medicaid",
            "UnitedHealthcare",
            "Dual Eligible",
        ],
        value="Medicare",
    ),
    gr.Number(label="Total Claim Cost", value=26483),
    gr.Number(label="Income", value=74739),
    gr.Dropdown(
        label="Admission Day of Week",
        choices=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        value="Tuesday",
    ),
    gr.Dropdown(
        label="Primary Diagnosis Code",
        choices=[
            "424132000",
            "6525002",
            "25675004",
            "183996000",
            "399261000",
            "74400008",
            "67811000119102",
            "39898005",
            "88805009",
            "698306007",
        ],
        value="424132000",
    ),
    gr.Dropdown(
        label="Provider ID",
        choices=[
            "us-npi|9999868992",
            "us-npi|9999975797",
            "us-npi|9999965897",
            "us-npi|9999897892",
            "us-npi|9999988691",
            "us-npi|9999948190",
            "us-npi|9999921791",
            "us-npi|9999936591",
            "us-npi|9999999490",
            "us-npi|9999978791",
        ],
        value="us-npi|9999868992",
    ),
    gr.Number(label="Prior Admissions (Last Year)", value=2),
    gr.Number(label="Number of Diagnoses", value=1),
    gr.Number(label="Number of Procedures", value=9),
    gr.Number(label="Number of Medications", value=1),
]

interface1 = gr.Interface(
    fn=interactive_prediction,
    inputs=interactive_inputs,
    outputs=gr.Markdown(label="Prediction Result"),
    title="Interactive Prediction",
    description=(
        "Enter synthetic encounter details to explore the demonstration model. "
        "Use a valid SNOMED CT code for the primary diagnosis.<br><br>"
        "**Demonstration only:** The 70% threshold is not clinically validated. "
        "Probabilities at or above the threshold are labeled High Risk; lower "
        "probabilities are labeled Low Risk."
    ),
)

# --- Interface for Tab 2 (ID-Based Prediction) ---
encounter_ids = list_inpatient_encounter_ids()

interface2 = gr.Interface(
    fn=id_based_prediction,
    inputs=gr.Dropdown(
        label="Encounter ID",
        choices=encounter_ids,
        value=encounter_ids[0] if encounter_ids else None,
        allow_custom_value=True,
    ),
    outputs=gr.Markdown(label="Prediction Result"),
    title="Predict from ID",
    description=(
        "Select an inpatient encounter from the generated database, or paste an "
        "encounter ID. Run the data pipeline first if no choices are available."
    ),
)

# --- Combine interfaces into a single app with tabs ---
app = gr.TabbedInterface(
    [interface1, interface2], ["Interactive Prediction", "Predict from ID"]
)

# --- Launch the app ---
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
