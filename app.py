import streamlit as st
import numpy as np
import pickle

with open('models/scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open('models/model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

st.title("E-Commerce Sales Predictor")

avg_session_length = st.number_input("Avg. Session Length")
time_on_app = st.number_input("Time on App")
length_of_membership = st.number_input("Length of Membership")


if st.button("Predict"):
    input_data = np.array([avg_session_length, time_on_app , length_of_membership]).reshape(1, -1)
    
    scaled_data = loaded_scaler.transform(input_data)
    
    prediction = loaded_model.predict(scaled_data)
    
    st.success(f" Yearly Amount Spent is : $ {prediction[0]}")