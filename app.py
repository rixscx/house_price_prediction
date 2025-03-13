import asyncio
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Fix event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load the trained model
model_path = "house_price_model.pkl"
if not os.path.exists(model_path):
    st.error("Model file not found! Please upload the model file.")
    model = None
else:
    with open(model_path, "rb") as file:
        model = pickle.load(file)

# Load dataset columns
csv_path = "california_housing.csv"
if not os.path.exists(csv_path):
    st.error("Dataset file not found! Please upload the dataset file.")
    columns = []
else:
    columns = pd.read_csv(csv_path, nrows=1).columns.tolist()

# Streamlit UI
st.set_page_config(page_title="California House Price Predictor", page_icon="🏡", layout="centered")

st.title("🏡 California House Price Prediction")
st.write("Fill in the details below to get a house price prediction.")

# Input fields
st.sidebar.header("Input Features")
input_data = {}
for col in columns:
    if col.lower() not in ["target", "medhousevalue"]:  # Exclude target column if present
        input_data[col] = st.sidebar.number_input(f"Enter {col}", value=0.0, step=0.01, format="%.2f")

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure input columns match model's expected features
if model:
    expected_features = model.get_booster().feature_names
    missing_features = [feat for feat in expected_features if feat not in input_df.columns]
    
    if missing_features:
        st.error(f"Missing required features: {missing_features}")
    else:
        # Select only the features expected by the model
        input_df = input_df[expected_features]

        # Prediction button
        if st.button("📊 Predict Price"):
            try:
                prediction = model.predict(input_df)
                st.success(f"🏠 Estimated House Price: **${prediction[0]:,.2f}**")
            except ValueError as e:
                st.error(f"Prediction error: {e}")
else:
    st.error("Model could not be loaded. Please check the model file.")

# Footer
st.markdown("---")
st.markdown("🔹 Developed with ❤️ using Streamlit")
