import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Diabetes Risk Assessment", layout="centered")

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

if __name__ == "__main__":
    main()
