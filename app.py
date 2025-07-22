
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Diabetes Predictor",
                   page_icon="🩺",
                   layout="centered")

# Custom CSS
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

def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Predictor</h1>', unsafe_allow_html=True)

    # Input form
    st.subheader("Enter Patient Details:")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120)

    model = load_model()

    if st.button("Predict"):
        if model:
            user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, diabetes_pedigree, age]])
            prediction = model.predict(user_input)[0]

            # Display result
            if prediction == 1:
                st.markdown('<div class="prediction-box diabetic">⚠️ <strong>Diabetic</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box not-diabetic">✅ <strong>Not Diabetic</strong></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
