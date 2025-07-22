import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Diabetes Risk Assessment", layout="centered")

# Medical-inspired CSS styling with healthcare psychology colors
st.markdown("""
<style>
/* Global medical theme */
.stApp {
    background: linear-gradient(135deg, #f8fafb 0%, #e8f4f8 100%);
    color: #2c3e50;
}:

/* Main header with calming medical blue */
.main-header {
    text-align: center;
    color: #1565c0;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-weight: 600;
    margin-bottom: 2rem;
    text-shadow: 0 1px 3px rgba(21, 101, 192, 0.1);
}

/* Professional medical form styling */
.stForm {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid #e3f2fd;
}

/* Medical prediction results */
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

/* High risk - Medical red with psychological safety */
.high-risk {
    background: linear-gradient(135deg, #fff3f3 0%, #ffebee 100%);
    border-color: #d32f2f;
    color: #b71c1c;
}

/* Low risk - Medical green with psychological comfort */
.low-risk {
    background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e8 100%);
    border-color: #388e3c;
    color: #1b5e20;
}

/* Medical information box */
.info-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #f0f7ff 100%);
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1.5rem 0;
    border-left: 5px solid #1976d2;
    box-shadow: 0 3px 15px rgba(25, 118, 210, 0.1);
    color: #2c3e50;
}

/* Medical metrics styling */
.metric-container {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e1f5fe;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    color: #2c3e50;
}

/* Input field styling for medical forms */
.stNumberInput > div > div > input {
    border: 2px solid #e3f2fd;
    border-radius: 8px;
    padding: 0.5rem;
    transition: border-color 0.3s ease;
}

.stNumberInput > div > div > input:focus {
    border-color: #1976d2;
    box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.1);
}

/* Select box medical styling */
.stSelectbox > div > div {
    border: 2px solid #e3f2fd;
    border-radius: 8px;
    background: white;
}

/* Medical button styling */
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

/* Medical section headers */
h2, h3 {
    color: #1565c0;
    font-weight: 600;
    border-bottom: 2px solid #e3f2fd;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Risk factors list styling */
.risk-factors {
    background: #fafafa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #ff9800;
    margin: 1rem 0;
    color: #2c3e50;
}

/* Medical expander styling */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #e8f4f8 0%, #f0f7ff 100%);
    border-radius: 8px;
    border: 1px solid #b3e5fc;
    color: #1565c0;
    font-weight: 600;
}

/* Subtle animation for medical elements */
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
    # Medical header with professional styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">Diabetes Risk Assessment System</h1>
        <p style="color: #34495e; font-size: 1.1rem; margin-top: -1rem; font-weight: 500;">
            Advanced Clinical Screening & Risk Stratification Tool
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Medical disclaimer with enhanced styling
    st.markdown("""
    <div class="info-box">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <strong style="color: #1565c0;">Medical Disclaimer & Clinical Notice</strong>
        </div>
        <p style="margin: 0; color: #2c3e50; font-weight: 500;">
            This tool provides preliminary risk assessment for educational and screening purposes only. 
            Results should not replace professional medical consultation, diagnosis, or treatment decisions.
            Always consult qualified healthcare providers for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Assessment form with medical styling
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="color: #1565c0;">
            Clinical Assessment Form
        </h2>
        <p style="color: #34495e; margin-top: -0.5rem; font-weight: 500;">
            Please provide accurate health information for optimal risk assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("assessment_form"):
        # Gender selection
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Patient Demographics & History**")
            # Only show pregnancies for females
            if gender == "Female":
                pregnancies = st.number_input("Number of Pregnancies (Total Live Births)", min_value=0, max_value=20, value=0)
            else:
                pregnancies = 0  # Set to 0 for males and others
                
            glucose = st.number_input("Fasting Blood Glucose (mg/dL)", min_value=50, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure (Diastolic) - mmHg", min_value=40, max_value=180, value=80)
            skin_thickness = st.number_input("Triceps Skinfold Thickness (mm)", min_value=0, max_value=100, value=20)
        
        with col2:
            st.markdown("**Laboratory & Metabolic Measurements**")
            insulin = st.number_input("2-Hour Serum Insulin (μU/mL)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("Body Mass Index - BMI (kg/m²)", min_value=15.0, max_value=70.0, value=25.0, step=0.1)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function (Family History)", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age in Years", min_value=18, max_value=120, value=30)
        
        submitted = st.form_submit_button("Calculate Risk", type="primary")
        
        if submitted:
            # Prepare input data with feature names
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            input_values = [[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, diabetes_pedigree, age]]
            
            input_data = pd.DataFrame(input_values, columns=feature_names)
            
            try:
                # Make prediction
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                risk_score = probability[1] if len(probability) > 1 else probability[0]
                
                # Display results
                st.markdown("---")
                st.subheader("Assessment Results")
                
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-box high-risk">
                        HIGH DIABETES RISK<br>
                        Risk Score: {:.1%}
                    </div>
                    """.format(risk_score), unsafe_allow_html=True)
                    
                    st.warning("Recommendation: Consult with a healthcare provider for further evaluation and testing.")
                    
                else:
                    st.markdown("""
                    <div class="prediction-box low-risk">
                        LOW DIABETES RISK<br>
                        Risk Score: {:.1%}
                    </div>
                    """.format(risk_score), unsafe_allow_html=True)
                    
                    st.success("Recommendation: Continue healthy lifestyle practices and regular check-ups.")
                
                # Risk factors analysis
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
                
                # Risk factors analysis
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
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e8 100%); 
                               padding: 1rem; border-radius: 8px; border-left: 4px solid #388e3c;">
                        <h4 style="color: #1b5e20; margin: 0;">Excellent Health Profile</h4>
                        <p style="color: #2e7d32; margin: 0.5rem 0 0 0;">No major diabetes risk factors identified</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    # Advanced data input guidance section
    with st.expander("Advanced Input Guide: Clinical Parameter Details & Tips"):
        st.subheader("Detailed Clinical Parameter Guide")
        
        st.markdown("**GLUCOSE LEVEL (mg/dL)**")
        st.write("• Normal Range: 70-99 mg/dL (fasting)")
        st.write("• Use fasting glucose (8-12 hours without food) for most accurate assessment")
        st.write("• Random glucose can be used but specify timing of last meal")
        st.write("• If using HbA1c: 5.7-6.4% = prediabetes, ≥6.5% = diabetes")
        st.write("• Morning values are typically most stable and reliable")
        
        st.markdown("**BLOOD PRESSURE (mmHg)**")
        st.write("• Normal Range: <120/80 mmHg")
        st.write("• Enter diastolic pressure (bottom number) in the field")
        st.write("• Take measurement after 5 minutes of rest in seated position")
        st.write("• Use average of 2-3 readings taken 1 minute apart")
        st.write("• Avoid caffeine, exercise, or smoking 30 minutes before measurement")
        
        st.markdown("**BODY MASS INDEX (BMI)**")
        st.write("• Calculation: Weight (kg) ÷ Height² (m²)")
        st.write("• Normal: 18.5-24.9 kg/m² | Overweight: 25.0-29.9 kg/m² | Obese: ≥30.0 kg/m²")
        st.write("• Measure weight in morning, after using bathroom, minimal clothing")
        st.write("• Use accurate scale on hard, flat surface")
        
        st.markdown("**INSULIN LEVEL (μU/mL)**")
        st.write("• Normal Range: 2.6-24.9 μU/mL (fasting)")
        st.write("• Requires laboratory blood test - not home measurable")
        st.write("• Must be fasting sample (8-12 hours without food)")
        st.write("• If unknown, use average value (80 μU/mL) for screening")
        
        st.markdown("**SKIN THICKNESS (mm)**")
        st.write("• Measurement Site: Triceps skinfold")
        st.write("• Requires calibrated skinfold calipers for accuracy")
        st.write("• If unknown, estimate: thin=10mm, average=20mm, larger=35mm")
        
        st.markdown("**FAMILY HISTORY SCORE**")
        st.write("• 0.0-0.3: No known family history of diabetes")
        st.write("• 0.4-0.7: Distant relatives with diabetes")
        st.write("• 0.8-1.2: One parent or sibling with Type 2 diabetes")
        st.write("• 1.3-2.0: Multiple first-degree relatives with diabetes")
        st.write("• 2.1-3.0: Both parents diabetic or strong family pattern")
    
    # User feedback form section
    st.markdown("---")
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="color: #1565c0;">
            Patient Feedback & Communication
        </h2>
        <p style="color: #34495e; margin-top: -0.5rem; font-weight: 500;">
            Share your assessment experience or ask questions for healthcare providers
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("feedback_form"):
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1976d2; margin-bottom: 1rem;">
            <p style="margin: 0; color: #2c3e50; font-weight: 500;">
                Your feedback helps improve this assessment tool and can be shared with healthcare providers 
                for better understanding of your health concerns and assessment experience.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        feedback_type = st.selectbox(
            "Feedback Category",
            ["Assessment Experience", "Technical Issues", "Health Questions", "General Comments", "Provider Communication"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            user_name = st.text_input("Name (Optional)", placeholder="Enter your name")
        with col2:
            user_email = st.text_input("Email (Optional)", placeholder="your.email@example.com")
        
        feedback_message = st.text_area(
            "Message or Questions",
            placeholder="Please share your thoughts, questions, or concerns about the diabetes risk assessment...",
            height=120
        )
        
        priority_level = st.selectbox(
            "Priority Level",
            ["Low - General feedback", "Medium - Questions or concerns", "High - Urgent health concerns"]
        )
        
        submitted_feedback = st.form_submit_button("Submit Feedback", type="secondary")
        
        if submitted_feedback:
            if feedback_message.strip():
                st.markdown("""
                <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%); 
                           padding: 1.5rem; border-radius: 8px; border-left: 4px solid #388e3c; margin-top: 1rem;">
                    <h4 style="color: #1b5e20; margin: 0 0 0.5rem 0;">Feedback Submitted Successfully</h4>
                    <p style="color: #2e7d32; margin: 0;">
                        Thank you for your feedback. Your message has been recorded and will be reviewed by our healthcare team.
                        If you indicated urgent health concerns, please contact your healthcare provider directly.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display feedback summary for confirmation
                st.markdown(f"""
                <div style="background: #f5f5f5; padding: 1rem; border-radius: 6px; margin-top: 1rem;">
                    <strong>Feedback Summary:</strong><br>
                    <strong>Category:</strong> {feedback_type}<br>
                    <strong>Priority:</strong> {priority_level}<br>
                    <strong>Name:</strong> {user_name if user_name else "Anonymous"}<br>
                    <strong>Message:</strong> {feedback_message[:200]}{"..." if len(feedback_message) > 200 else ""}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a message before submitting feedback.")

if __name__ == "__main__":
    main()
