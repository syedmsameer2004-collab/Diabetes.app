import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Diabetes Risk Assessment", layout="centered")

# Medical-inspired CSS styling with healthcare psychology colors
st.markdown("""
<style>
/* Global medical theme - Dark Mode */
.stApp {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d30 100%);
    color: #e8e8e8;
}

/* Main header with calming medical blue */
.main-header {
    text-align: center;
    color: #1565c0;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-weight: 600;
    margin-bottom: 2rem;
    text-shadow: 0 1px 3px rgba(21, 101, 192, 0.1);
}

/* Professional medical form styling - Dark Mode */
.stForm {
    background: #2d2d30;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    border: 1px solid #404040;
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

/* Medical information box - Dark Mode */
.info-box {
    background: linear-gradient(135deg, #2a2a2a 0%, #353535 100%);
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1.5rem 0;
    border-left: 5px solid #1976d2;
    box-shadow: 0 3px 15px rgba(25, 118, 210, 0.3);
    color: #e8e8e8;
}

/* Medical metrics styling - Dark Mode */
.metric-container {
    background: #2d2d30;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #404040
