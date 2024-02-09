import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load the pre-trained Gradient Boosting model
model_gb = joblib.load('gradient_boosting_model.pkl')

# Create a function to make predictions based on user inputs


def predict_price(city, oem, transmission, fuel_type, ownership):
    # Create a DataFrame with user inputs
    user_input = pd.DataFrame({
        'City': [city],
        'OEM': [oem],
        'Transmission': [transmission],
        'Fuel_Type': [fuel_type],
        'Ownership': [ownership]
    })

    # Add more features as needed

    # Perform any necessary preprocessing (e.g., encoding, feature engineering)

    # Make predictions
    predicted_price = model_gb.predict(user_input)

    return predicted_price


# Streamlit app
st.title("Car Price Prediction App")

# User inputs
city = st.selectbox("Select City", [
                    "Bangalore", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata"])
oem = st.selectbox("Select OEM", ['Maruti', 'Ford', 'Tata', 'Hyundai', 'Jeep', 'Datsun', 'Honda', 'Mahindra', 'BMW', 'Renault', 'Mercedes-Benz', 'Audi', 'Mini', 'Kia', 'Skoda', 'Volkswagen', 'Volvo', 'MG', 'Toyota', 'Nissan',
                                  'Mahindra Ssangyong', 'Mitsubishi', 'Jaguar', 'Fiat', 'Land Rover',
                                  'Chevrolet', 'Citroen', 'Mahindra Renault', 'Isuzu', 'Lexus',
                                  'Porsche'])  # Add actual OEM names
transmission = st.selectbox("Select Transmission", ["Manual", "Automatic"])
fuel_type = st.selectbox("Select Fuel Type", [
                         "Petrol", "Diesel", "LPG", "CNG", "Electric"])
ownership = st.selectbox("Select Ownership", [
                         "First Owner", "Second Owner", "Third Owner", "Fourth Owner", "Fifth Owner"])

# Add more features as needed

# Predict car price
predicted_price = predict_price(city, oem, transmission, fuel_type, ownership)

# Display the predicted price
st.write(f"Predicted Car Price: {predicted_price[0]:,.2f} INR")
