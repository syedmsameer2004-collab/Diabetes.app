
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic diabetes dataset for training"""
    np.random.seed(42)

    
    pregnancies = np.random.poisson(2, n_samples)
    glucose = np.random.normal(120, 30, n_samples)
    blood_pressure = np.random.normal(80, 15, n_samples)
    skin_thickness = np.random.normal(20, 10, n_samples)
    insulin = np.random.normal(80, 40, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    diabetes_pedigree = np.random.gamma(0.5, 1, n_samples)
    age = np.random.normal(35, 15, n_samples)

    
    pregnancies = np.clip(pregnancies, 0, 17)
    glucose = np.clip(glucose, 50, 300)
    blood_pressure = np.clip(blood_pressure, 40, 180)
    skin_thickness = np.clip(skin_thickness, 0, 100)
    insulin = np.clip(insulin, 0, 900)
    bmi = np.clip(bmi, 15, 70)
    diabetes_pedigree = np.clip(diabetes_pedigree, 0, 3)
    age = np.clip(age, 18, 120)

    
    diabetes_risk = (
        0.1 * (glucose > 125) +
        0.08 * (bmi > 30) +
        0.06 * (age > 45) +
        0.05 * (blood_pressure > 90) +
        0.04 * diabetes_pedigree +
        0.03 * (pregnancies > 3) +
        np.random.normal(0, 0.1, n_samples)
    )

    outcome = (diabetes_risk > 0.3).astype(int)

    
    data = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age,
        'Outcome': outcome
    })

    return data

def train_and_save_model():
    """Train the diabetes prediction model and save it"""

    
    data = generate_synthetic_data(1000)

    
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )

    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    
    joblib.dump(model, 'diabetes_model.pkl')
    print("\nModel saved as 'diabetes_model.pkl'")

    return model

if __name__ == "__main__":
    train_and_save_model()
