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
    
