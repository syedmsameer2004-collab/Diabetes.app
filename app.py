import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}
.prediction-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    text-align: center;
}
.diabetic {
    background-color: #ffebee;
    border: 2px solid #f44336;
    color: #d32f2f;
}
.not-diabetic {
    background-color: #e8f5e8;
    border: 2px solid #4caf50;
    color: #2e7d32;
}
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'diabetes_model.pkl' not found!")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_status_icon(value, metric_type):
    if metric_type == 'glucose':
        if value < 70:
            return "Low"
        elif value <= 100:
            return "Normal"
        elif value <= 125:
            return "Elevated"
        else:
            return "High"
    elif metric_type == 'blood_pressure':
        if value < 80:
            return "Normal"
        elif value <= 89:
            return "Elevated"
        else:
            return "High"
    elif metric_type == 'bmi':
        if value < 18.5:
            return "Underweight"
        elif value < 25:
            return "Normal"
        elif value < 30:
            return "Overweight"
        else:
            return "Obese"
    elif metric_type == 'age':
        if value < 30:
            return "Young"
        elif value < 50:
            return "Adult"
        else:
            return "Senior"
    else:
        return "Recorded"

def main():
    # Header
    st.markdown('<h1 class="main-header">Diabetes Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #2c5aa0; font-size: 18px; margin-bottom: 2rem; font-weight: 500;">
    This app predicts the risk of diabetes based on your health data.
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.stop()

    show_prediction_page(model)

def show_prediction_page(model):
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 80)
            skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        with col2:
            insulin = st.number_input("Insulin Level (mu U/ml)", 0, 900, 80)
            bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.001)
            age = st.number_input("Age (years)", 1, 120, 30)

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
            try:
                prediction = model.predict(input_data)
                if prediction[0] == 1:
                    st.error("ELEVATED DIABETES RISK - High probability of diabetes detected")
                else:
                    st.success("LOW DIABETES RISK - Current health indicators within normal ranges")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
