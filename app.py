import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

# UI for user input
st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict the risk of heart disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.radio("Sex", [0, 1])  # 0: Female, 1: Male
cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)", [0, 1])
restecg = st.radio("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.radio("Exercise Induced Angina (1 = Yes; 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.radio("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.radio("Thalassemia (0-3)", [0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("The patient is at risk of heart disease. 🚨")
    else:
        st.success("The patient is not at risk of heart disease. ✅")