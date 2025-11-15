import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Load Model and Scalers 

model = joblib.load('Models Notebooks/knn_final_model.joblib')
feature_scaler = joblib.load('Models Notebooks/feature_scaler.joblib')
target_scaler = joblib.load('Models Notebooks/target_scaler.joblib')


# --- Feature Engineering Function ---
def create_time_features(dt):
    """Generates the time-based features required by the model."""
    
    # 1. Date and Time Features
    hour = dt.hour + dt.minute / 60
    day = dt.day
    month = dt.month
    
    # 2. Cyclical Features
    features = {
        'HourCos': np.cos(2 * np.pi * hour / 24),
        'HourSin': np.sin(2 * np.pi * hour / 24),
        'MonthCos': np.cos(2 * np.pi * month / 12),
        'MonthSin': np.sin(2 * np.pi * month / 12),
        'DayCos': np.cos(2 * np.pi * day / 31),
        'DaySin': np.sin(2 * np.pi * day / 31),
        
        # 3. Categorical/Constant Features
        'Year': 2017, # Constant based on training data
        'IsWeekend': 1 if dt.weekday() >= 5 else 0, # Saturday or Sunday
        'IsNight': 1 if dt.hour < 7 or dt.hour >= 21 else 0 # Example: 9 PM to 7 AM
    }
    return features

# --- Streamlit App Interface ---
st.title("⚡️ Power Consumption Prediction (KNN)")
st.markdown("Predict the total power consumption for the entire city.")

# --- Input Fields ---
with st.container(border=True):
    st.subheader("Time & Date")
    
    col1, col2 = st.columns(2)
    with col1:
        date_input = st.date_input("Date")
    with col2:
        time_input = st.time_input("Time")

    st.subheader("Weather Conditions")
    
    # Input for continuous weather features
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=15.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=10.0, value=2.0)
    general_diffuse_flows = st.number_input("General Diffuse Flows", min_value=0.0, max_value=1500.0, value=500.0)
    diffuse_flows = st.number_input("Diffuse Flows", min_value=0.0, max_value=1500.0, value=200.0)

# --- Prediction Logic ---
if st.button("Predict Consumption"):
    # Combine date and time inputs into a single datetime object
    dt = pd.to_datetime(f"{date_input} {time_input}")
    
    # Generate time-based features
    time_features = create_time_features(dt)
    
    # Prepare the raw input DataFrame in the exact order the model expects
    raw_input = {
        'Temperature': temp,
        'Humidity': humidity,
        'Wind Speed': wind_speed,
        'general diffuse flows': general_diffuse_flows,
        'diffuse flows': diffuse_flows,
        # The next 9 features are the engineered/categorical ones
        'Year': time_features['Year'],
        'IsWeekend': time_features['IsWeekend'],
        'IsNight': time_features['IsNight'],
        'HourCos': time_features['HourCos'],
        'HourSin': time_features['HourSin'],
        'MonthCos': time_features['MonthCos'],
        'MonthSin': time_features['MonthSin'],
        'DayCos': time_features['DayCos'],
        'DaySin': time_features['DaySin']
    }
    
    # Create the DataFrame
    input_df = pd.DataFrame([raw_input])
    
    # Scale features using the fitted feature scaler
    scaled_input = feature_scaler.transform(input_df)
    
    # Make prediction (output is scaled)
    scaled_prediction = model.predict(scaled_input)
    
    # Inverse transform to get the prediction in original units
    final_prediction = target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0][0]
    
    # Display Result
    st.success(f"Predicted Total City Consumption: {final_prediction:,.2f} kWh")