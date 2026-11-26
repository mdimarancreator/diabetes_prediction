
import os
from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

import joblib
import pandas as pd
import streamlit as st
from google import genai
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# Gemini AI Configuration
# --------------------------------------------------

@st.cache_resource
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


try:
    gemini_client = get_gemini_client()
except Exception:
    gemini_client = None


GEMINI_MODEL = "gemini-3.1-flash-lite"


def ask_gemini(prompt):
   
    """Generate an educational response using Gemini AI."""

    if gemini_client is None:
        return "ERROR: Gemini client is not initialized. Check your API key configuration."

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text or "ERROR: Gemini returned an empty response."

    except Exception as e:
        st.error(f"Gemini API Error: {type(e).__name__}: {e}")
        return "Gemini could not generate a response. Check the error above."


    except Exception:
        return (
            "Gemini AI is temporarily unavailable. "
            "Please try again later."
        )




# --------------------------------------------------
# PDF Report Generator
# --------------------------------------------------

def create_prediction_pdf(prediction, ai_explanation=None):
    """Build a PDF report for the current diabetes risk prediction."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="AI Diabetes Risk Prediction Report"
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"],
        alignment=TA_CENTER, textColor=colors.HexColor("#1565C0"),
        fontSize=20, leading=25, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle", parent=styles["Heading2"],
        textColor=colors.HexColor("#087E8B"),
        fontSize=13, leading=17, spaceBefore=12, spaceAfter=7
    ))
    styles.add(ParagraphStyle(
        name="SmallDisclaimer", parent=styles["BodyText"],
        fontSize=8, leading=11, textColor=colors.HexColor("#555555")
    ))

    probability = float(prediction["probability"])
    risk_label = (str(prediction["result_text"])
                  .replace("✅", "").replace("⚠️", "")
                  .replace("🚨", "").strip())

    story = [
        Paragraph("AI Diabetes Risk Predictor", styles["ReportTitle"]),
        Paragraph("Machine Learning Prediction Report", styles["Heading2"]),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["SmallDisclaimer"]
        ),
        Spacer(1, 7 * mm),
        Paragraph("Prediction Summary", styles["SectionTitle"])
    ]

    summary = [
        ["Model-estimated probability", f"{probability:.1f}%"],
        ["Risk category", risk_label],
        ["Algorithm", "Random Forest Classifier"]
    ]
    summary_table = Table(summary, colWidths=[75 * mm, 85 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3FB")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DEE8")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8)
    ]))
    story.append(summary_table)

    story.append(Paragraph("Submitted Health Parameters", styles["SectionTitle"]))
    labels = {
        "Pregnancies": "Pregnancies",
        "Glucose": "Glucose (mg/dL)",
        "BloodPressure": "Blood Pressure (mm Hg)",
        "SkinThickness": "Skin Thickness (mm)",
        "Insulin": "Insulin (mu U/ml)",
        "BMI": "Body Mass Index (BMI)",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
        "Age": "Age (years)"
    }
    values = prediction["input_data"].iloc[0].to_dict()
    rows = [["Parameter", "Entered value"]]
    for key, label in labels.items():
        value = values.get(key, "N/A")
        if isinstance(value, float):
            value = f"{value:.2f}"
        rows.append([label, str(value)])

    table = Table(rows, colWidths=[90 * mm, 70 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DEE8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F5F8FB")]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8)
    ]))
    story.append(table)

    story.append(Paragraph("General Recommendation", styles["SectionTitle"]))
    story.append(Paragraph(escape(str(prediction["recommendation"])),
                           styles["BodyText"]))

    if ai_explanation and not str(ai_explanation).startswith("ERROR:"):
        story.append(Paragraph("Gemini AI Explanation", styles["SectionTitle"]))
        safe_text = escape(str(ai_explanation)).replace("\n", "<br/>")
        story.append(Paragraph(safe_text, styles["BodyText"]))

    story.extend([
        Spacer(1, 7 * mm),
        Paragraph("Medical Disclaimer", styles["SectionTitle"]),
        Paragraph(
            "This report is an educational estimate generated by a machine "
            "learning model. It is not a clinical test, medical diagnosis, "
            "or treatment recommendation. The displayed probability may not "
            "be clinically calibrated. Do not use this result to start, stop, "
            "or change medication. Consult a qualified healthcare professional "
            "for appropriate testing and medical advice.",
            styles["SmallDisclaimer"]
        )
    ])

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


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
# Load Machine Learning Model
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
    'An intelligent diabetes risk prediction system powered by '
    'Machine Learning and Generative AI'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("ℹ️ Project Information")

    st.write("**ML Algorithm:** Random Forest Classifier")
    st.write("**AI Model:** Google Gemini")
    st.write("**Dataset:** Pima Indians Diabetes Dataset")
    st.write("**Framework:** Streamlit")
    st.write("**Libraries:** Pandas, Scikit-learn, Joblib, Google GenAI")

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
# AI Privacy Notice and Consent
# --------------------------------------------------

st.info(
    "Privacy notice: If you enable AI explanations, the health "
    "values entered below will be sent to Google's Gemini API. "
    "Chatbot messages are also processed by Gemini."
)

allow_ai_explanation = st.checkbox(
    "I agree to share these health values with Gemini to generate an AI explanation."
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

    st.session_state.pop("ai_explanation", None)

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

    probability = model.predict_proba(input_data)[0][1] * 100

    if probability < 30:
        css_class = "low"
        result_text = "✅ Low Risk"
        recommendation = (
            "Your model-estimated risk is relatively low. "
            "Continue healthy lifestyle habits and routine checkups."
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
            "Please consult a qualified healthcare professional."
        )

    # Store prediction results
    st.session_state["prediction"] = {
        "probability": probability,
        "css_class": css_class,
        "result_text": result_text,
        "recommendation": recommendation,
        "input_data": input_data
    }


# --------------------------------------------------
# Display Prediction Results
# --------------------------------------------------

if "prediction" in st.session_state:

    result = st.session_state["prediction"]

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.metric(
            label="Model-Estimated Risk",
            value=f"{result['probability']:.1f}%"
        )

    with right_column:
        st.markdown(
            f'<div class="result-box {result["css_class"]}">'
            f'{result["result_text"]}'
            '</div>',
            unsafe_allow_html=True
        )

    st.progress(int(result["probability"]))

    st.write(result["recommendation"])

    with st.expander("📋 View Submitted Health Information"):
        st.dataframe(
            result["input_data"],
            use_container_width=True
        )

    # Reports are provided in PDF format only.
    pdf_data = create_prediction_pdf(
        result,
        st.session_state.get("ai_explanation")
    )
    st.download_button(
        label="📄 Download Prediction Report (PDF)",
        data=pdf_data,
        file_name="diabetes_prediction_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # --------------------------------------------------
    # Gemini AI Prediction Explanation
    # --------------------------------------------------

    st.markdown("---")
    st.subheader("🤖 AI Prediction Explanation")

    if allow_ai_explanation:

        if st.button("✨ Generate AI Explanation"):

            values = result["input_data"].iloc[0].to_dict()

            prompt = f"""
            You are an educational health information assistant.

            Explain the following machine learning prediction
            in simple and professional English.

            Model-estimated probability: {result['probability']:.1f}%
            Model risk category: {result['result_text']}

            Input health parameters:
            {values}

            Instructions:
            1. Explain what this model output means.
            2. Briefly describe the input parameters.
            3. Suggest general healthy lifestyle practices.
            4. Recommend professional medical evaluation when appropriate.
            5. Do not diagnose diabetes.
            6. Do not prescribe medication.
            7. Do not change or reinterpret the model prediction.
            8. Clearly mention that this is not a clinical assessment.
            """

            with st.spinner("Gemini is preparing your explanation..."):
                explanation = ask_gemini(prompt)

            st.session_state["ai_explanation"] = explanation
            st.rerun()

        if "ai_explanation" in st.session_state:
            st.write(st.session_state["ai_explanation"])

    else:
        st.warning(
            "Please enable the AI explanation consent checkbox "
            "before sending health information to Gemini."
        )

    st.caption(
        "Disclaimer: This prediction is an educational estimate "
        "generated by a machine learning model. It does not "
        "confirm or rule out diabetes."
    )



# --------------------------------------------------
# Gemini AI Health Assistant - Modern Chat Interface
# --------------------------------------------------

st.markdown("""
<style>
.chat-header {
    background: linear-gradient(120deg, #1565c0, #00897b);
    padding: 24px;
    border-radius: 18px;
    color: white;
    margin-bottom: 20px;
}

.chat-header h2 {
    color: white;
    margin: 0;
    font-size: 27px;
}

.chat-header p {
    color: #eafaf8;
    margin-top: 8px;
    margin-bottom: 0;
    font-size: 14px;
}

.chat-note {
    background: rgba(128, 128, 128, 0.10);
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 13px;
    margin-bottom: 15px;
}

div[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 12px;
}

div[data-testid="stChatInput"] {
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="chat-header">
    <h2>✨ Gemini AI Health Assistant</h2>
    <p>Your intelligent companion for diabetes awareness and healthy living.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Chat Session Management
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# --------------------------------------------------
# Chat Toolbar
# --------------------------------------------------

toolbar_col1, toolbar_col2 = st.columns([4, 1])

with toolbar_col1:
    st.markdown("### 💬 Conversation")

with toolbar_col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()


st.markdown("""
<div class="chat-note">
    🔒 <b>Privacy:</b> Avoid sharing your name, contact details,
    medical records, or other personal information.
    Messages are processed by Google's Gemini API.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Welcome Message
# --------------------------------------------------

if not st.session_state["chat_history"]:

    with st.chat_message("assistant", avatar="✨"):
        st.markdown("""
        Hello! 👋 Welcome to your **Gemini AI Health Assistant**.

        I can help you understand:

        - 🩺 Diabetes and its risk factors
        - 🥗 Healthy food and nutrition
        - 🏃 Exercise and physical activity
        - 💡 Diabetes prevention and awareness

        **How can I help you today?**
        """)


# --------------------------------------------------
# Suggested Questions
# --------------------------------------------------

if not st.session_state["chat_history"]:

    st.markdown("**Try asking:**")

    suggestion_col1, suggestion_col2 = st.columns(2)

    with suggestion_col1:
        suggestion_one = st.button(
            "🩺 What is diabetes?",
            use_container_width=True
        )

        suggestion_two = st.button(
            "🥗 Healthy food for diabetes",
            use_container_width=True
        )

    with suggestion_col2:
        suggestion_three = st.button(
            "🏃 Best exercises for health",
            use_container_width=True
        )

        suggestion_four = st.button(
            "💡 How can diabetes be prevented?",
            use_container_width=True
        )

    if suggestion_one:
        st.session_state["pending_question"] = "What is diabetes?"

    elif suggestion_two:
        st.session_state["pending_question"] = (
            "What are some healthy food choices for people with diabetes?"
        )

    elif suggestion_three:
        st.session_state["pending_question"] = (
            "What physical activities support good health?"
        )

    elif suggestion_four:
        st.session_state["pending_question"] = (
            "How can the risk of type 2 diabetes be reduced?"
        )


# --------------------------------------------------
# Display Chat History
# --------------------------------------------------

for message in st.session_state["chat_history"]:

    avatar = "✨" if message["role"] == "assistant" else "👤"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# --------------------------------------------------
# User Message Input
# --------------------------------------------------

user_question = st.chat_input(
    "Ask Gemini about diabetes, nutrition, or health..."
)

if "pending_question" in st.session_state:
    user_question = st.session_state.pop("pending_question")


# --------------------------------------------------
# Gemini Response Generation
# --------------------------------------------------

if user_question:

    st.session_state["chat_history"].append({
        "role": "user",
        "content": user_question
    })

    chatbot_prompt = f"""
    You are a professional educational health awareness assistant.

    Answer the user's question in simple, clear and friendly English.

    Guidelines:
    - Explain diabetes, nutrition, exercise and prevention.
    - Provide general educational information.
    - Use headings and bullet points when helpful.
    - Never diagnose a medical condition.
    - Never prescribe medication or treatment.
    - Encourage professional medical consultation when appropriate.
    - For emergencies, advise immediate medical care.

    User question:
    {user_question}
    """

    with st.chat_message("assistant", avatar="✨"):

        with st.spinner("Gemini is preparing your answer..."):

            answer = ask_gemini(chatbot_prompt)

        st.markdown(answer)

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()


# --------------------------------------------------
# Chatbot Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Powered by Google Gemini AI | "
    "Educational purposes only — Not a substitute for medical advice."
)
