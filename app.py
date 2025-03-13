import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset columns
columns = pd.read_csv("california_housing.csv", nrows=1).columns.tolist()

st.title("California Housing Price Prediction")
st.write("Enter the required details to predict house price.")

# Create input fields for each feature
input_data = {}
for col in columns:
    if col != "target":  # Exclude target column if present
        input_data[col] = st.number_input(f"Enter {col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
