
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Diabetes Risk Assessment", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d30 100%);
    color: #e8e8e8;
}
.main-header {
    text-align: center;
    color: #1565c0;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-weight: 600;
    margin-bottom: 2rem;
    text-shadow: 0 1px 3px rgba(21, 101, 192, 0.1);
}
.stForm {
    background: #2d2d30;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    border: 1px solid #404040;
}
.prediction-box {
    padding: 2rem;
    border-radius: 12px;
    margin: 1.5rem 0;
    text-align: center;
    font-weight: 600;
    font-size: 1.3rem;
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.12);
    border: 2px solid;
}
.high-risk {
    background: linear-gradient(135deg, #fff3f3 0%, #ffebee 100%);
    border-color: #d32f2f;
    color: #b71c1c;
}
.low-risk {
    background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e8 100%);
    border-color: #388e3c;
    color: #1b5e20;
}
.info-box {
    background: linear-gradient(135deg, #2a2a2a 0%, #353535 100%);
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1.5rem 0;
    border-left: 5px solid #1976d2;
    box-shadow: 0 3px 15px rgba(25, 118, 210, 0.3);
    color: #e8e8e8;
}
.metric-container {
    background: #2d2d30;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #404040;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    color: #e8e8e8;
}
.stSelectbox > div > label,
.stNumberInput > div > label,
.stTextInput > div > label,
.stTextArea > div > label {
    color: #81d4fa !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
.stNumberInput > div > div > input {
    border: 2px solid #404040;
    border-radius: 8px;
    padding: 0.5rem;
    transition: border-color 0.3s ease;
    background: #2d2d30;
    color: #e8e8e8;
}
.stNumberInput > div > div > input:focus {
    border-color: #1976d2;
    box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.3);
}
.stSelectbox > div > div {
    border: 2px solid #404040;
    border-radius: 8px;
    background: #2d2d30;
    color: #e8e8e8;
}
.stSelectbox > div > label {
    color: #81d4fa !important;
    font-weight: 600 !important;
}
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);
    transition: all 0.3s ease;
}
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
    box-shadow: 0 6px 20px rgba(25, 118, 210, 0.4);
    transform: translateY(-2px);
}
h2, h3 {
    color: #1565c0;
    font-weight: 600;
    border-bottom: 2px solid #e3f2fd;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.stMarkdown strong,
.stMarkdown b {
    color: #81d4fa !important;
    font-weight: 600 !important;
}
.risk-factors {
    background: #2a2a2a;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #ff9800;
    margin: 1rem 0;
    color: #e8e8e8;
}
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #2a2a2a 0%, #353535 100%);
    border-radius: 8px;
    border: 1px solid #404040;
    color: #64b5f6;
    font-weight: 600;
}
@keyframes medicalPulse {
    0% { opacity: 0.9; }
    50% { opacity: 1; }
    100% { opacity: 0.9; }
}
.prediction-box {
    animation: medicalPulse 2s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

def main():
    st.title("Diabetes Risk Assessment Tool")
    model = load_model()
    if model is None:
        st.stop()

    with st.form("assessment_form"):
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0) if gender == "Female" else 0
        glucose = st.number_input("Fasting Blood Glucose", min_value=50, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=180, value=80)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("2-Hour Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=15.0, max_value=70.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=18, max_value=120, value=30)

        submitted = st.form_submit_button("Calculate Risk")

        if submitted:
            features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            values = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
            input_data = pd.DataFrame(values, columns=features)

            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                risk_score = probability[1] if len(probability) > 1 else probability[0]
                st.metric("Risk Score", f"{risk_score:.2%}")
                if prediction == 1:
                    st.error("High Risk of Diabetes")
                else:
                    st.success("Low Risk of Diabetes")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()

    st.markdown("---")
    st.subheader("Feedback Form")
    with st.form("feedback_form"):
        feedback_type = st.selectbox(
            "Feedback Category",
            ["Assessment Experience", "Technical Issues", "Health Questions", "General Comments", "Provider Communication"]
        )
        name = st.text_input("Name (Optional)")
        email = st.text_input("Email (Optional)")
        message = st.text_area("Your Feedback", height=120)
        priority = st.selectbox("Priority Level", ["Low", "Medium", "High"])
        submitted_feedback = st.form_submit_button("Submit Feedback")

        if submitted_feedback:
            if message.strip():
                st.success("Thank you for your feedback!")
                st.markdown(f"**Category:** {feedback_type}")
                st.markdown(f"**Priority:** {priority}")
                st.markdown(f"**Name:** {name or 'Anonymous'}")
                st.markdown(f"**Message:** {message}")
            else:
                st.warning("Please enter a message before submitting.")
