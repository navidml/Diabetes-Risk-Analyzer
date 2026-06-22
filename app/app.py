import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="DiaPredict AI",
    layout="wide"
)
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#0ea5e9,#2563eb);
    color: white;
    border-radius: 12px;
    border: none;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#0284c7,#1d4ed8);
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

.title-box {
    text-align:center;
    padding:20px;
    background: linear-gradient(135deg,#0ea5e9,#2563eb);
    border-radius:15px;
    color:white;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# Load Model
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "diabetes_prediction_model_v2.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()
# ==================================================
# Header
# ==================================================

st.markdown("""
<div class="title-box">
    <h1>🩺 DiaPredict AI</h1>
    <h3>Intelligent Diabetes Risk Prediction System</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
This AI-powered application predicts the likelihood of diabetes
based on patient clinical information.
""")

# ==================================================
# Input Section
# ==================================================

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0,
        max_value=500,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=300,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:

    insulin = st.number_input(
        "Insulin Level",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

# ==================================================
# Prediction
# ==================================================

if st.button("Predict Diabetes Risk"):

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    probability = model.predict_proba(input_data)[0][1]

    threshold = 0.35

    prediction = 1 if probability >= threshold else 0

    st.markdown("---")

    if prediction == 1:

        st.error("High Diabetes Risk Detected")

        st.metric(
            "Predicted Risk",
            f"{probability * 100:.2f}%"
        )

        st.warning(
            "This prediction suggests an elevated risk of diabetes."
        )

    else:

        st.success("Low Diabetes Risk")

        st.metric(
            "Predicted Risk",
            f"{probability * 100:.2f}%"
        )

        st.info(
            "The prediction indicates a relatively low risk of diabetes."
        )
    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability*100,
    title={'text': "Diabetes Risk"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range':[0,35], 'color':"lightgreen"},
            {'range':[35,70], 'color':"gold"},
            {'range':[70,100], 'color':"salmon"}
        ]
    }
))

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# Sidebar
# ==================================================

st.sidebar.title("About DiaPredict AI")

st.sidebar.markdown("""
### Diabetes Prediction System

This application uses a trained machine learning model to estimate diabetes risk based on patient health indicators.

### Features

- Diabetes Risk Prediction
- Probability Estimation
- Clinical Data Analysis
- Fast AI Inference

### Technologies

- Python
- Scikit-Learn
- Joblib
- Streamlit
""")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Performance")

st.sidebar.write("Accuracy: 82.3%")
st.sidebar.write("Recall: 83.1%")
st.sidebar.write("F1 Score: 76.6%")
st.sidebar.write("MCC: 0.63")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Educational and research purposes only. "
    "This tool is not a substitute for professional medical advice."
)

