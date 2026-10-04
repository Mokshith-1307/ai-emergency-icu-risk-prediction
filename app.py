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

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "timing_corrected_icu_model.pkl"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "anchor_age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2",
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
# LOGIN
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login_page():

    st.title("🏥 Hospital Clinical Decision Support")

    st.subheader("Authorized Hospital Access")

    st.info(
        "This prototype is intended for authorized "
        "hospital/clinical personnel."
    )

    username = st.text_input(
        "Hospital Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "🔐 Login",
        use_container_width=True
    ):

        # DEMO credentials only.
        # For real deployment, use Streamlit Secrets.

        if (
            username == "hospital_admin"
            and password == "Hospital@123"
        ):

            st.session_state.logged_in = True

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )


# ============================================================
# STOP HERE IF NOT LOGGED IN
# ============================================================

if not st.session_state.logged_in:

    login_page()

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_FILE
    )


try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained ICU prediction model."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🏥 AI-Driven Emergency Patient ICU Risk Prediction"
)

st.write(
    "Clinical Decision-Support Prototype for "
    "Early ICU Admission Risk Estimation"
)

st.success(
    "Authorized hospital user logged in"
)


# ============================================================
# LOGOUT
# ============================================================

if st.button("Logout"):

    st.session_state.logged_in = False

    st.rerun()


st.divider()


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:

    patient_id = st.text_input(
        "Patient / Case ID"
    )

with col2:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=120,
        value=60
    )

with col3:

    sex = st.selectbox(
        "Sex",
        [
            "Not specified",
            "Male",
            "Female"
        ]
    )


# ============================================================
# VITAL SIGNS
# ============================================================

st.header("🫀 Vital Signs")

col1, col2, col3 = st.columns(3)

with col1:

    heart_rate = st.number_input(
        "Heart Rate (bpm)",
        min_value=30.0,
        max_value=220.0,
        value=80.0
    )

with col2:

    systolic_bp = st.number_input(
        "Systolic BP (mmHg)",
        min_value=50.0,
        max_value=250.0,
        value=120.0
    )

with col3:

    diastolic_bp = st.number_input(
        "Diastolic BP (mmHg)",
        min_value=30.0,
        max_value=150.0,
        value=80.0
    )


col1, col2 = st.columns(2)

with col1:

    respiratory_rate = st.number_input(
        "Respiratory Rate (/min)",
        min_value=5.0,
        max_value=60.0,
        value=18.0
    )

with col2:

    spo2 = st.number_input(
        "SpO₂ (%)",
        min_value=50.0,
        max_value=100.0,
        value=98.0
    )


# ============================================================
# LABORATORY VALUES
# ============================================================

st.header("🧪 Laboratory Results")

st.caption(
    "Enter available laboratory values. "
    "The trained model uses median imputation "
    "for missing values."
)


col1, col2, col3 = st.columns(3)

with col1:

    creatinine = st.number_input(
        "Creatinine",
        min_value=0.0,
        max_value=15.0,
        value=1.0
    )

with col2:

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )

with col3:

    potassium = st.number_input(
        "Potassium",
        min_value=1.0,
        max_value=10.0,
        value=4.0
    )


col1, col2, col3 = st.columns(3)

with col1:

    sodium = st.number_input(
        "Sodium",
        min_value=100.0,
        max_value=200.0,
        value=138.0
    )

with col2:

    chloride = st.number_input(
        "Chloride",
        min_value=50.0,
        max_value=150.0,
        value=102.0
    )

with col3:

    bicarbonate = st.number_input(
        "Bicarbonate",
        min_value=5.0,
        max_value=50.0,
        value=24.0
    )


col1, col2, col3 = st.columns(3)

with col1:

    hemoglobin = st.number_input(
        "Hemoglobin",
        min_value=3.0,
        max_value=25.0,
        value=12.0
    )

with col2:

    wbc = st.number_input(
        "WBC",
        min_value=0.0,
        max_value=150.0,
        value=8.0
    )

with col3:

    platelets = st.number_input(
        "Platelets",
        min_value=1.0,
        max_value=1000.0,
        value=250.0
    )


col1, col2 = st.columns(2)

with col1:

    albumin = st.number_input(
        "Albumin",
        min_value=1.0,
        max_value=7.0,
        value=4.0
    )

with col2:

    lactate = st.number_input(
        "Lactate",
        min_value=0.0,
        max_value=20.0,
        value=1.0
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

patient_data = {
    "anchor_age": age,
    "heart_rate": heart_rate,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "respiratory_rate": respiratory_rate,
    "spo2": spo2,
    "creatinine": creatinine,
    "glucose": glucose,
    "potassium": potassium,
    "sodium": sodium,
    "chloride": chloride,
    "bicarbonate": bicarbonate,
    "hemoglobin": hemoglobin,
    "wbc": wbc,
    "platelets": platelets,
    "albumin": albumin,
    "lactate": lactate
}


# ============================================================
# CLINICAL ABNORMALITY CHECK
# ============================================================

def detect_abnormalities(data):

    abnormalities = []

    if data["spo2"] < 90:

        abnormalities.append(
            "Low oxygen saturation (possible hypoxemia)"
        )

    if data["systolic_bp"] < 90:

        abnormalities.append(
            "Low systolic blood pressure (possible hypotension)"
        )

    if data["systolic_bp"] >= 180:

        abnormalities.append(
            "Markedly elevated systolic blood pressure"
        )

    if data["heart_rate"] > 100:

        abnormalities.append(
            "Tachycardia (elevated heart rate)"
        )

    elif data["heart_rate"] < 60:

        abnormalities.append(
            "Bradycardia (low heart rate)"
        )

    if data["respiratory_rate"] > 20:

        abnormalities.append(
            "Tachypnea (elevated respiratory rate)"
        )

    elif data["respiratory_rate"] < 12:

        abnormalities.append(
            "Low respiratory rate"
        )

    if data["creatinine"] > 1.5:

        abnormalities.append(
            "Elevated creatinine (possible renal dysfunction)"
        )

    if data["glucose"] > 250:

        abnormalities.append(
            "Marked hyperglycemia"
        )

    elif data["glucose"] < 70:

        abnormalities.append(
            "Hypoglycemia"
        )

    if data["potassium"] > 5.5:

        abnormalities.append(
            "Hyperkalemia"
        )

    elif data["potassium"] < 3.5:

        abnormalities.append(
            "Hypokalemia"
        )

    if data["sodium"] > 145:

        abnormalities.append(
            "Hypernatremia"
        )

    elif data["sodium"] < 135:

        abnormalities.append(
            "Hyponatremia"
        )

    if data["bicarbonate"] < 18:

        abnormalities.append(
            "Low bicarbonate (possible metabolic acidosis)"
        )

    if data["hemoglobin"] < 8:

        abnormalities.append(
            "Significant anemia"
        )

    if data["wbc"] > 20:

        abnormalities.append(
            "Markedly elevated WBC"
        )

    elif data["wbc"] < 4:

        abnormalities.append(
            "Low WBC"
        )

    if data["platelets"] < 100:

        abnormalities.append(
            "Thrombocytopenia (low platelet count)"
        )

    if data["albumin"] < 3:

        abnormalities.append(
            "Low albumin"
        )

    if data["lactate"] > 2:

        abnormalities.append(
            "Elevated lactate"
        )

    return abnormalities


# ============================================================
# PREDICTION
# ============================================================

st.divider()

st.header("🔬 ICU Risk Prediction")

if st.button(
    "🚨 Predict ICU Admission Risk",
    type="primary",
    use_container_width=True
):

    X = pd.DataFrame(
        [patient_data]
    )

    X = X[FEATURES]

    try:

        probability = model.predict_proba(
            X
        )[0][1]

        prediction = model.predict(
            X
        )[0]

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.code(
            str(e)
        )

        st.stop()


    probability_percent = (
        probability * 100
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if probability < 0.33:

        risk_level = "LOW"

    elif probability < 0.66:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"


    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    st.subheader(
        "Prediction Result"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "ICU Admission Probability",
            f"{probability_percent:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Level",
            risk_level
        )

    with col3:

        st.metric(
            "Model Classification",
            "Higher Risk"
            if prediction == 1
            else "Lower Risk"
        )


    if risk_level == "HIGH":

        st.error(
            "🔴 HIGH RISK — Early clinical assessment "
            "and physician review are recommended."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "🟠 MEDIUM RISK — Clinical monitoring "
            "and physician review are recommended."
        )

    else:

        st.success(
            "🟢 LOW RISK — Continue appropriate "
            "clinical monitoring."
        )


    # ========================================================
    # POSSIBLE CLINICAL PROBLEMS
    # ========================================================

    st.subheader(
        "🩺 Possible Clinical Problems / Abnormalities"
    )

    abnormalities = detect_abnormalities(
        patient_data
    )

    if abnormalities:

        for abnormality in abnormalities:

            st.warning(
                "• " + abnormality
            )

    else:

        st.success(
            "No major rule-based abnormality "
            "detected from the entered values."
        )


    st.caption(
        "These are screening indicators based on "
        "predefined thresholds and are NOT definitive diagnoses."
    )


    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

    st.subheader(
        "📋 Patient Summary"
    )

    summary = pd.DataFrame({
        "Parameter": [
            "Patient / Case ID",
            "Age",
            "Sex",
            "Heart Rate",
            "Systolic BP",
            "Diastolic BP",
            "Respiratory Rate",
            "SpO₂",
            "ICU Risk"
        ],

        "Value": [
            patient_id if patient_id else "Not provided",
            f"{age} years",
            sex,
            f"{heart_rate:.1f} bpm",
            f"{systolic_bp:.1f} mmHg",
            f"{diastolic_bp:.1f} mmHg",
            f"{respiratory_rate:.1f} /min",
            f"{spo2:.1f}%",
            f"{probability_percent:.2f}% ({risk_level})"
        ]
    })

    st.table(
        summary
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ Clinical Decision-Support Disclaimer

    This application is a research prototype and is not a
    substitute for professional medical diagnosis or treatment.

    The prediction represents an estimated ICU admission risk
    based on the machine-learning model and entered clinical data.

    Possible clinical abnormalities are screening indicators and
    should not be interpreted as definitive diagnoses.

    Final triage, treatment, and ICU admission decisions must be
    made by qualified healthcare professionals.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "AI-Driven Emergency Patient Triage and Early ICU Admission Prediction "
    "| Research Prototype"
)