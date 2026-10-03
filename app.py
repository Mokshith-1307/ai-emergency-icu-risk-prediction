import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Emergency ICU Risk Prediction",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_FILE = BASE_DIR / "models" / "robust_icu_risk_model.pkl"


# ============================================================
# TITLE
# ============================================================

st.title("🏥 AI-Driven Emergency Patient ICU Risk Prediction")

st.markdown(
    """
    ### Clinical Decision-Support Prototype

    This application uses machine learning to estimate the
    likelihood of ICU admission from selected clinical
    laboratory measurements.

    **Important:** This is a research prototype and is not a
    replacement for clinical judgment.
    """
)

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "creatinine",
    "glucose",
    "potassium",
    "sodium",
    "chloride",
    "bicarbonate",
    "hemoglobin",
    "wbc",
    "platelets",
    "albumin",
    "lactate"
]
# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    return model


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Patient Information")

st.sidebar.info(
    "Enter the available laboratory measurements. "
    "The trained machine-learning model will estimate "
    "ICU admission risk."
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("patient_form"):

    st.subheader("🧪 Clinical Laboratory Measurements")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        creatinine = st.number_input(
            "Creatinine",
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=0.1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=1.0
        )

        potassium = st.number_input(
            "Potassium",
            min_value=0.0,
            max_value=15.0,
            value=4.0,
            step=0.1
        )

        sodium = st.number_input(
            "Sodium",
            min_value=50.0,
            max_value=200.0,
            value=140.0,
            step=1.0
        )

    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        chloride = st.number_input(
            "Chloride",
            min_value=50.0,
            max_value=200.0,
            value=100.0,
            step=1.0
        )

        bicarbonate = st.number_input(
            "Bicarbonate",
            min_value=0.0,
            max_value=100.0,
            value=24.0,
            step=1.0
        )

        hemoglobin = st.number_input(
            "Hemoglobin",
            min_value=0.0,
            max_value=30.0,
            value=13.0,
            step=0.1
        )

        wbc = st.number_input(
            "WBC",
            min_value=0.0,
            max_value=500.0,
            value=8.0,
            step=0.1
        )

    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        platelets = st.number_input(
            "Platelets",
            min_value=0.0,
            max_value=2000.0,
            value=250.0,
            step=1.0
        )

        albumin = st.number_input(
            "Albumin",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1
        )

        lactate = st.number_input(
            "Lactate",
            min_value=0.0,
            max_value=30.0,
            value=1.5,
            step=0.1
        )

    submitted = st.form_submit_button(
        "🔍 Predict ICU Admission Risk"
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # CREATE PATIENT DATAFRAME
    # --------------------------------------------------------

    patient_data = pd.DataFrame(
        [[
            creatinine,
            glucose,
            potassium,
            sodium,
            chloride,
            bicarbonate,
            hemoglobin,
            wbc,
            platelets,
            albumin,
            lactate
        ]],
        columns=FEATURES
    )


    # --------------------------------------------------------
    # PREDICT PROBABILITY
    # --------------------------------------------------------

    probability = model.predict_proba(
        patient_data
    )[0][1]


    # --------------------------------------------------------
    # PREDICT CLASS
    # --------------------------------------------------------

    prediction = int(
        probability >= 0.50
    )


    # --------------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------------

    if probability < 0.33:

        risk_level = "LOW"

    elif probability < 0.66:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "ICU Admission Probability",
            f"{probability * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Predicted Class",
            "ICU Admission"
            if prediction == 1
            else "No ICU Admission"
        )


    with col3:

        st.metric(
            "Risk Level",
            risk_level
        )


    # --------------------------------------------------------
    # DECISION SUPPORT
    # --------------------------------------------------------

    st.subheader("🚨 Decision-Support Alert")


    if risk_level == "HIGH":

        st.error(
            "HIGH RISK: The model estimates a high probability "
            "of ICU admission. Immediate clinical assessment "
            "is recommended."
        )


    elif risk_level == "MEDIUM":

        st.warning(
            "MEDIUM RISK: The model estimates an intermediate "
            "probability of ICU admission. Further clinical "
            "assessment is recommended."
        )


    else:

        st.success(
            "LOW RISK: The model estimates a lower probability "
            "of ICU admission based on the entered laboratory data."
        )


    # --------------------------------------------------------
    # INPUT SUMMARY
    # --------------------------------------------------------

    st.subheader("🧾 Patient Input Summary")

    st.dataframe(
        patient_data,
        width="stretch"
    )
# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Research prototype only. The prediction is intended for "
    "academic decision-support research and must not be used "
    "as a standalone clinical decision."
)