import os
import joblib
import pandas as pd
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.main-title {
    text-align: center; font-size: 42px; font-weight: 800;
    background: linear-gradient(90deg, #1e88e5, #00c6a7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sub-title { text-align: center; color: #888; font-size: 18px; margin-bottom: 25px; }
.result-box {
    padding: 25px; border-radius: 16px; text-align: center;
    color: white; font-size: 26px; font-weight: 700; margin-top: 10px;
}
.low  { background: linear-gradient(135deg, #2e7d32, #66bb6a); }
.mid  { background: linear-gradient(135deg, #ef6c00, #ffb74d); }
.high { background: linear-gradient(135deg, #c62828, #ef5350); }
.stButton > button {
    width: 100%; height: 52px; font-size: 20px; font-weight: 700;
    border-radius: 12px; background: linear-gradient(90deg, #1e88e5, #00c6a7);
    color: white; border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
if not os.path.exists(model_path):
    st.error("model.pkl nahi mili. Pehle `python train.py` chalaiye.")
    st.stop()
model = joblib.load(model_path)

# ---------- Header ----------
st.markdown('<div class="main-title">🩺 AI Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning se diabetes ka early risk check kijiye</div>', unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ Project Info")
    st.write("**Model:** Random Forest")
    st.write("**Dataset:** Pima Indians Diabetes")
    st.write("**Tools:** Python, Pandas, Scikit-learn, Streamlit")
    st.info("Ye tool sirf awareness ke liye hai, medical diagnosis nahi. Doctor se zaroor milein.")

# ---------- Inputs ----------
st.subheader("📝 Apni details bhariye")
c1, c2, c3, c4 = st.columns(4)
with c1:
    preg = st.number_input("🤰 Pregnancies", 0, 20, 0)
    skin = st.number_input("📏 Skin Thickness (mm)", 0, 100, 20)
with c2:
    glu = st.number_input("🍬 Glucose (mg/dL)", 0, 300, 100)
    ins = st.number_input("💉 Insulin (mu U/ml)", 0, 900, 80)
with c3:
    bp = st.number_input("❤️ Blood Pressure (mm Hg)", 0, 200, 70)
    bmi = st.number_input("⚖️ BMI", 0.0, 70.0, 25.0)
with c4:
    dpf = st.number_input("🧬 Diabetes Pedigree", 0.0, 3.0, 0.5)
    age = st.number_input("🎂 Age", 1, 120, 30)

st.write("")
predict = st.button("🔍 Predict Risk")

# ---------- Prediction ----------
if predict:
    data = pd.DataFrame([[preg, glu, bp, skin, ins, bmi, dpf, age]],
        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

    prob = model.predict_proba(data)[0][1] * 100

    if prob < 30:
        css, text, tip = "low", "✅ Low Risk", "Aapka risk kam hai. Healthy lifestyle jaari rakhiye."
    elif prob < 60:
        css, text, tip = "mid", "⚠️ Medium Risk", "Risk thoda hai. Khane aur exercise par dhyan dijiye, checkup karwaiye."
    else:
        css, text, tip = "high", "🚨 High Risk", "Risk zyada hai. Kripya jaldi doctor se milein."

    st.markdown("---")
    st.subheader("📊 Result")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Diabetes Risk", f"{prob:.1f} %")
    with right:
        st.markdown(f'<div class="result-box {css}">{text}</div>', unsafe_allow_html=True)
    st.progress(int(prob))
    st.write(tip)

    with st.expander("📋 Aapki di hui values dekhiye"):
        st.dataframe(data, use_container_width=True)

    st.caption("⚠️ Ye sirf awareness ke liye hai, medical diagnosis nahi.")