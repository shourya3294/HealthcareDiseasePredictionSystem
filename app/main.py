import streamlit as st
import pandas as pd
import pickle
import os

# Load models
model_paths = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[name] = pickle.load(f)

st.title("Healthcare Disease Prediction System")

st.write("Predict the risk of diseases such as Diabetes and Heart Disease.")

# User input
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)

if st.button("Predict"):
    input_data = [[age, bmi, blood_pressure]]
    for name, model in models.items():
        prediction = model.predict(input_data)[0]
        st.write(f"{name} Prediction: {'Disease Risk' if prediction == 1 else 'No Disease Risk'}")
