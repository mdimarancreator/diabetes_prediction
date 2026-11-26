
import os
import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS Styling
# --------------------------------------------------

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #1e88e5, #00c6a7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    text-align: center;
    color: #888;
    font-size: 18px;
    margin-bottom: 25px;
}

.result-box {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    color: white;
    font-size: 26px;
    font-weight: 700;
    margin-top: 10px;
}

.low {
    background: linear-gradient(135deg, #2e7d32, #66bb6a);
}

.mid {
    background: linear-gradient(135deg, #ef6c00, #ffb74d);
}

.high {
    background: linear-gradient(135deg, #c62828, #ef5350);
}

.stButton > button {
    width: 100%;
    height: 52px;
    font-size: 20px;
    font-weight: 700;
    border-radius: 12px;
    background: linear-gradient(90deg, #1e88e5, #00c6a7);
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load the Trained Machine Learning Model
# --------------------------------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")

if not os.path.exists(model_path):
    st.error("Model file not found. Please run train.py first.")
    st.stop()

model = joblib.load(model_path)


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🩺 AI Diabetes Risk Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'An intelligent diabetes risk prediction system powered by Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar - Project Information
# --------------------------------------------------

with st.sidebar:
    st.header("ℹ️ Project Information")

    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Dataset:** Pima Indians Diabetes Dataset")
    st.write("**Framework:** Streamlit")
    st.write("**Libraries:** Pandas, Scikit-learn, Joblib")

    st.info(
        "This application is designed for educational and "
        "health awareness purposes only. It is not a substitute "
        "for professional medical advice or diagnosis."
    )


# --------------------------------------------------
# User Input Section
# --------------------------------------------------

st.subheader("📝 Enter Your Health Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    pregnancies = st.number_input(
        "🤰 Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

    skin_thickness = st.number_input(
        "📏 Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:
    glucose = st.number_input(
        "🍬 Glucose Level (mg/dL)",
        min_value=0,
        max_value=300,
        value=100
    )

    insulin = st.number_input(
        "💉 Insulin Level (mu U/ml)",
        min_value=0,
        max_value=900,
        value=80
    )

with col3:
    blood_pressure = st.number_input(
        "❤️ Blood Pressure (mm Hg)",
        min_value=0,
        max_value=200,
        value=70
    )

    bmi = st.number_input(
        "⚖️ Body Mass Index (BMI)",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

with col4:
    diabetes_pedigree = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "🎂 Age (Years)",
        min_value=1,
        max_value=120,
        value=30
    )


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

st.write("")

predict_button = st.button("🔍 Predict Diabetes Risk")


# --------------------------------------------------
# Generate Prediction
# --------------------------------------------------

if predict_button:

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Calculate the predicted probability
    probability = model.predict_proba(input_data)[0][1] * 100

    # Classify the risk level
    if probability < 30:
        css_class = "low"
        result_text = "✅ Low Risk"
        recommendation = (
            "Your model-estimated risk is relatively low. "
            "Continue maintaining healthy lifestyle habits "
            "and attend routine health checkups."
        )

    elif probability < 60:
        css_class = "mid"
        result_text = "⚠️ Medium Risk"
        recommendation = (
            "Your model-estimated risk is moderate. "
            "Consider discussing your health parameters "
            "with a qualified healthcare professional."
        )

    else:
        css_class = "high"
        result_text = "🚨 High Risk"
        recommendation = (
            "Your model-estimated risk is relatively high. "
            "Please consult a qualified healthcare professional "
            "for proper evaluation and appropriate testing."
        )

    # Display prediction results
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.metric(
            label="Model-Estimated Risk",
            value=f"{probability:.1f}%"
        )

    with right_column:
        st.markdown(
            f'<div class="result-box {css_class}">'
            f'{result_text}'
            '</div>',
            unsafe_allow_html=True
        )

    st.progress(int(probability / 100 * 100))

    st.write(recommendation)

    # Display submitted health information
    with st.expander("📋 View Submitted Health Information"):
        st.dataframe(
            input_data,
            use_container_width=True
        )

    st.caption(
        "Disclaimer: This prediction is an educational estimate "
        "generated by a machine learning model. It does not "
        "confirm or rule out diabetes."
    )
