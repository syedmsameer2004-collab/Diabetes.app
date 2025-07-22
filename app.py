import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Diabetes Predictor",
                   page_icon="🩺",
                   layout="centered")

# Custom CSS with Medical Psychology-Based Design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        text-align: center;
        color: #2c5aa0;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(44, 90, 160, 0.1);
        background: linear-gradient(90deg, #2c5aa0, #4a90e2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Medical Card Styling */
    .medical-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(44, 90, 160, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    
    /* Form Styling */
    .stForm {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 
            inset 5px 5px 10px #e6e9f0,
            inset -5px -5px 10px #ffffff;
        border: 2px solid rgba(44, 90, 160, 0.08);
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input {
        background: linear-gradient(145deg, #f8f9ff, #ffffff);
        border: 2px solid #e1e8f5;
        border-radius: 12px;
        padding: 12px 16px;
        color: #2c5aa0;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4a90e2 0%, #2c5aa0 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(44, 90, 160, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(44, 90, 160, 0.4);
        background: linear-gradient(135deg, #5ba0f2 0%, #3c6ab0 100%);
    }
    
    /* Success Alert - Calming Green */
    .success-alert {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f9f0 100%);
        border: 2px solid #81c784;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #2e7d32;
        box-shadow: 0 4px 20px rgba(129, 199, 132, 0.2);
    }
    
    /* Error Alert - Warm Warning Orange */
    .error-alert {
        background: linear-gradient(135deg, #fff3e0 0%, #ffeaa7 100%);
        border: 2px solid #ff9800;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #e65100;
        box-shadow: 0 4px 20px rgba(255, 152, 0, 0.2);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f0f9ff 100%);
        border: 2px solid #64b5f6;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #1565c0;
        box-shadow: 0 4px 20px rgba(100, 181, 246, 0.2);
    }
    
    /* Subheaders */
    .stSubheader {
        color: #2c5aa0;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9ff 0%, #e3f2fd 100%);
        border-radius: 12px;
        color: #2c5aa0;
        font-weight: 500;
    }
    
    /* Medical Icons and Decorations */
    .medical-icon {
        font-size: 2rem;
        color: #4a90e2;
        margin-right: 0.5rem;
    }
    
    /* Pulse Animation for Critical Results */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 152, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0); }
    }
    
    .pulse-warning {
        animation: pulse 2s infinite;
    }
    
    /* Soft Shadow for Elements */
    .soft-shadow {
        box-shadow: 0 2px 15px rgba(44, 90, 160, 0.08);
    }
    
    /* Column Spacing */
    .stColumn {
        padding: 0 1rem;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(44, 90, 160, 0.1);
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
    """Return status icon based on health metric values"""
    if metric_type == 'glucose':
        if value < 70:
            return "🔽 Low"
        elif value <= 100:
            return "✅ Normal"
        elif value <= 125:
            return "⚠️ Elevated"
        else:
            return "🔴 High"
    elif metric_type == 'blood_pressure':
        if value < 80:
            return "✅ Normal"
        elif value <= 89:
            return "⚠️ Elevated"
        else:
            return "🔴 High"
    elif metric_type == 'bmi':
        if value < 18.5:
            return "🔽 Underweight"
        elif value < 25:
            return "✅ Normal"
        elif value < 30:
            return "⚠️ Overweight"
        else:
            return "🔴 Obese"
    elif metric_type == 'age':
        if value < 30:
            return "👶 Young"
        elif value < 50:
            return "👨 Adult"
        else:
            return "👴 Senior"
    else:
        return "📊 Recorded"

def main():
    # Header with enhanced medical design
    st.markdown('<h1 class="main-header">🏥 Advanced Diabetes Risk Assessment</h1>',
                unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; color: #2c5aa0; font-size: 18px; margin-bottom: 2rem; font-weight: 500;">
            Professional medical screening tool powered by machine learning
        </div>
    """, unsafe_allow_html=True)

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    show_prediction_page(model)

def show_prediction_page(model):
    # Create input form with enhanced styling
    with st.form("prediction_form"):
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <h3 style="color: #2c5aa0; font-weight: 600; margin: 0;">
                    📋 Patient Health Assessment
                </h3>
                <p style="color: #666; margin: 0.5rem 0; font-size: 14px;">
                    Please provide accurate medical information for precise risk evaluation
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Number of Pregnancies",
                                          min_value=0,
                                          max_value=20,
                                          value=1,
                                          step=1)

            glucose = st.number_input("Glucose Level (mg/dL)",
                                      min_value=0,
                                      max_value=300,
                                      value=120,
                                      step=1)

            blood_pressure = st.number_input("Blood Pressure (mm Hg)",
                                             min_value=0,
                                             max_value=200,
                                             value=80,
                                             step=1)

            skin_thickness = st.number_input("Skin Thickness (mm)",
                                             min_value=0,
                                             max_value=100,
                                             value=20,
                                             step=1)

        with col2:
            insulin = st.number_input("Insulin Level (mu U/ml)",
                                      min_value=0,
                                      max_value=900,
                                      value=80,
                                      step=1)

            bmi = st.number_input("BMI (Body Mass Index)",
                                  min_value=0.0,
                                  max_value=70.0,
                                  value=25.0,
                                  step=0.1)

            diabetes_pedigree = st.number_input("Diabetes Pedigree Function",
                                                min_value=0.0,
                                                max_value=3.0,
                                                value=0.5,
                                                step=0.001,
                                                format="%.3f")

            age = st.number_input("Age (years)",
                                  min_value=1,
                                  max_value=120,
                                  value=30,
                                  step=1)

        # Submit button with enhanced styling
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔬 Analyze Risk Profile", use_container_width=True)

        if submitted:
            # Prepare input data as numpy array
            input_data = np.array([[
                pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                bmi, diabetes_pedigree, age
            ]])

            try:
                # Make prediction
                prediction = model.predict(input_data)

                # Display result with enhanced medical styling
                st.markdown("<br>", unsafe_allow_html=True)
                
                if prediction[0] == 1:
                    st.markdown("""
                        <div class="error-alert pulse-warning" style="text-align: center;">
                            <h2 style="color: #e65100; margin: 0; font-size: 1.8rem;">
                                ⚠️ ELEVATED DIABETES RISK
                            </h2>
                            <p style="font-size: 1.2rem; margin: 0.5rem 0; font-weight: 500;">
                                High probability of diabetes detected
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                        <div style="background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%); 
                                   border-radius: 12px; padding: 1.5rem; margin: 1rem 0; 
                                   border-left: 5px solid #ff9800;">
                            <h4 style="color: #e65100; margin-top: 0;">🏥 Immediate Medical Actions Required:</h4>
                            <ul style="color: #bf360c; line-height: 1.6;">
                                <li><strong>Schedule immediate consultation</strong> with healthcare provider</li>
                                <li><strong>Begin glucose monitoring</strong> as directed by physician</li>
                                <li><strong>Implement diabetic dietary plan</strong> with professional guidance</li>
                                <li><strong>Start supervised exercise program</strong> per medical recommendations</li>
                                <li><strong>Regular follow-up appointments</strong> for ongoing monitoring</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="success-alert" style="text-align: center;">
                            <h2 style="color: #2e7d32; margin: 0; font-size: 1.8rem;">
                                ✅ LOW DIABETES RISK
                            </h2>
                            <p style="font-size: 1.2rem; margin: 0.5rem 0; font-weight: 500;">
                                Current health indicators within normal ranges
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                                   border-radius: 12px; padding: 1.5rem; margin: 1rem 0; 
                                   border-left: 5px solid #4caf50;">
                            <h4 style="color: #2e7d32; margin-top: 0;">🌿 Preventive Health Measures:</h4>
                            <ul style="color: #1b5e20; line-height: 1.6;">
                                <li><strong>Maintain balanced nutrition</strong> with whole foods and controlled portions</li>
                                <li><strong>Regular physical activity</strong> - minimum 150 minutes weekly</li>
                                <li><strong>Weight management</strong> within healthy BMI range</li>
                                <li><strong>Annual health screenings</strong> for early detection</li>
                                <li><strong>Stress management</strong> and adequate sleep (7-9 hours)</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)

                # Show input summary with enhanced styling
                with st.expander("📊 View Detailed Health Profile"):
                    input_df = pd.DataFrame({
                        'Health Metric': [
                            '🤰 Pregnancies', '🩸 Glucose Level', '💓 Blood Pressure',
                            '📏 Skin Thickness', '💉 Insulin Level', '⚖️ BMI',
                            '🧬 Genetic Factor', '📅 Age'
                        ],
                        'Value': [
                            pregnancies, f"{glucose} mg/dL", f"{blood_pressure} mmHg",
                            f"{skin_thickness} mm", f"{insulin} μU/mL", f"{bmi:.1f}",
                            f"{diabetes_pedigree:.3f}", f"{age} years"
                        ],
                        'Status': [
                            get_status_icon(pregnancies, 'pregnancies'),
                            get_status_icon(glucose, 'glucose'),
                            get_status_icon(blood_pressure, 'blood_pressure'),
                            get_status_icon(skin_thickness, 'skin_thickness'),
                            get_status_icon(insulin, 'insulin'),
                            get_status_icon(bmi, 'bmi'),
                            get_status_icon(diabetes_pedigree, 'pedigree'),
                            get_status_icon(age, 'age')
                        ]
                    })
                    st.dataframe(input_df, use_container_width=True, hide_index=True)

                # Medical disclaimer with enhanced styling
                st.markdown("""
                    <div class="info-box" style="text-align: center;">
                        <h4 style="color: #1565c0; margin-top: 0;">🏥 Medical Disclaimer</h4>
                        <p style="color: #0d47a1; margin: 0; font-size: 14px; line-height: 1.5;">
                            This AI-powered assessment is for informational purposes only and should not replace 
                            professional medical diagnosis. Always consult qualified healthcare providers for 
                            accurate medical evaluation and treatment decisions.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
