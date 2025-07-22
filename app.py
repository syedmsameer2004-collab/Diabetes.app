import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Diabetes Risk Assessment", layout="centered")

# Medical-inspired CSS styling with better input label visibility
st.markdown("""
<style>  
/* Global medical theme - Dark Mode */  
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

/* Highlight labels clearly above input fields */  
label, .stSelectbox label, .stNumberInput label, .stTextInput label {  
    color: #64b5f6 !important;  
    font-weight: 700 !important;  
    font-size: 1rem !important;  
}  
  
/* Rest remains the same... */
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
    animation: medicalPulse 2s ease-in-out infinite;
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
.stNumberInput > div > div > input {
    border: 2px solid #404040;
    border-radius: 8px;
    padding: 0.5rem;
    background: #2d2d30;
    color: #e8e8e8;
}
.stNumberInput > div > div > input:focus {
    border-color: #1976d2;
    box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.3);
}
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);
}
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
    box-shadow: 0 6px 20px rgba(25, 118, 210, 0.4);
    transform: translateY(-2px);
}
@keyframes medicalPulse {
    0% { opacity: 0.9; }
    50% { opacity: 1; }
    100% { opacity: 0.9; }
}
</style>  
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

def main():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
    <h1 class="main-header">Diabetes Risk Assessment System</h1>
    <p style="color: #b0b0b0; font-size: 1.1rem; margin-top: -1rem; font-weight: 500;">
    Advanced Clinical Screening & Risk Stratification Tool
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong style="color: #1565c0;">Medical Disclaimer & Clinical Notice</strong>
        <p style="color: #e3f2fd; font-weight: 500;">
            This tool provides preliminary risk assessment for educational and screening purposes only. 
            Results should not replace professional medical consultation, diagnosis, or treatment decisions. 
            Always consult qualified healthcare providers for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.stop()

    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="color: #1565c0;">Clinical Assessment Form</h2>
        <p style="color: #b0b0b0; margin-top: -0.5rem; font-weight: 500;">
            Please provide accurate health information for optimal risk assessment
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("assessment_form"):
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Personal Information**")
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0) if gender == "Female" else 0
            glucose = st.number_input("Blood Sugar Level", min_value=50, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=180, value=80)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

        with col2:
            st.markdown("**Health Measurements**")
            insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=15.0, max_value=70.0, value=25.0, step=0.1)
            diabetes_pedigree = st.number_input("Family History Score", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age", min_value=18, max_value=120, value=30)

        submitted = st.form_submit_button("Calculate Risk", type="primary")

        if submitted:
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                        insulin, bmi, diabetes_pedigree, age]], columns=feature_names)

            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                risk_score = probability[1] if len(probability) > 1 else probability[0]

                st.markdown("---")
                st.subheader("Assessment Results")

                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box high-risk">
                        HIGH DIABETES RISK<br>
                        Risk Score: {risk_score:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("Recommendation: Consult with a healthcare provider for further evaluation and testing.")
                else:
                    st.markdown(f"""
                    <div class="prediction-box low-risk">
                        LOW DIABETES RISK<br>
                        Risk Score: {risk_score:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("Recommendation: Continue healthy lifestyle practices and regular check-ups.")

                st.subheader("Clinical Health Indicators")
                col1, col2, col3 = st.columns(3)

                with col1:
                    glucose_status = "HIGH" if glucose > 125 else "NORMAL" if glucose < 100 else "ELEVATED"
                    glucose_color = "#d32f2f" if glucose > 125 else "#388e3c" if glucose < 100 else "#ff9800"
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: {glucose_color}; margin: 0;">Glucose Level</h4>
                        <p style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">{glucose} mg/dL</p>
                        <p style="color: {glucose_color}; margin: 0; font-weight: 600;">{glucose_status}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    bmi_status = "HIGH" if bmi >= 30 else "NORMAL" if bmi < 25 else "ELEVATED"
                    bmi_color = "#d32f2f" if bmi >= 30 else "#388e3c" if bmi < 25 else "#ff9800"
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: {bmi_color}; margin: 0;">Body Mass Index</h4>
                        <p style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">{bmi:.1f} kg/m²</p>
                        <p style="color: {bmi_color}; margin: 0; font-weight: 600;">{bmi_status}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    bp_status = "HIGH" if blood_pressure > 90 else "NORMAL"
                    bp_color = "#d32f2f" if blood_pressure > 90 else "#388e3c"
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: {bp_color}; margin: 0;">Blood Pressure</h4>
                        <p style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">{blood_pressure} mmHg</p>
                        <p style="color: {bp_color}; margin: 0; font-weight: 600;">{bp_status}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.subheader("Risk Factor Analysis")
                risk_factors = []
                if glucose > 125:
                    risk_factors.append("Elevated glucose levels (Hyperglycemia)")
                if bmi >= 30:
                    risk_factors.append("Obesity (BMI ≥30)")
                elif bmi >= 25:
                    risk_factors.append("Overweight (BMI 25-29.9)")
                if blood_pressure > 90:
                    risk_factors.append("Hypertension (High blood pressure)")
                if age > 45:
                    risk_factors.append("Advanced age (>45 years)")
                if diabetes_pedigree > 0.5:
                    risk_factors.append("Genetic predisposition (Family history)")
                if gender == "Female" and pregnancies > 0:
                    risk_factors.append("Gestational diabetes risk factor")

                if risk_factors:
                    st.markdown("""
                    <div class="risk-factors">
                        <h4 style="color: #d32f2f; margin-bottom: 1rem;">Identified Risk Factors:</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for factor in risk_factors:
                        st.markdown(f"• **{factor}**")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
