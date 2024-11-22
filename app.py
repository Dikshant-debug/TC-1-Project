import streamlit as st
import numpy as np
import pickle

# Set the page title and favicon
st.set_page_config(page_title="Revenue Radar", page_icon="📊")

# Title of the app
st.title("📊 Revenue Radar")
st.subheader("Predict Yearly Customer Spending")

# Load the scaler and model using pickle
try:
    with open('models/scaler.pkl', 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)
    with open('models/model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Error: Model files not found! Ensure 'scaler.pkl' and 'model.pkl' exist in the 'models' directory.")
    st.stop()

# Input fields for user data
st.markdown("### Enter Customer Data:")
avg_session_length = st.number_input("💻 Average Session Length (minutes)", min_value=0.0, step=0.1)
time_on_app = st.number_input("📱 Time on App (minutes)", min_value=0.0, step=0.1)
length_of_membership = st.number_input("🗓️ Length of Membership (years)", min_value=0.0, step=0.1)

# Prediction button and logic
if st.button("🔍 Predict"):
    try:
        # Preparing input data
        input_data = np.array([avg_session_length, time_on_app, length_of_membership]).reshape(1, -1)
        
        # Scaling the data
        scaled_data = loaded_scaler.transform(input_data)
        
        # Making predictions
        prediction = loaded_model.predict(scaled_data)
        
        # Displaying the result
        st.success(f"💵 Predicted Yearly Amount Spent: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
