import streamlit as st
import joblib
import pandas as pd

# Load the trained model and threshold
model_file = "xgboost_tuned_model.pkl"
model_data = joblib.load(model_file)
model = model_data["model"]
best_threshold = model_data["best_threshold"]

# Title of the app
st.title('Accident Severity Prediction')

# Description
st.write("""
    **Accident Severity Prediction Tool**
    
    This tool predicts the likelihood of a **severe accident** based on various factors such as the time of day, road conditions, weather, and others.
    Please fill in the inputs below and click on **"Predict Accident Severity"** to get the result.

    **Disclaimer:**
    - The model was trained using **2023 accident data from the UK**. It may not be fully applicable to other regions or more recent data.
    - **Caution is advised** in making decisions based solely on the model’s prediction. Always exercise care and use judgment when planning your commute.
    - The tool provides predictions about the **likelihood** of an accident severity based on historical data and should not be relied upon for real-time risk assessment.
""")

# Feature Inputs with human-friendly interaction

# Urban or Rural Area Selection (Radio button)
urban_or_rural = st.radio(
    "Select the area type:",
    ["Urban", "Rural"],
    index=0  # Default to Urban
)

urban_or_rural_area = 1 if urban_or_rural == "Urban" else 2

# Weekend or Weekday Selection (Checkbox)
is_weekend = st.checkbox('Is it a weekend?', value=False)
is_weekend = 1 if is_weekend else 0

# Peak Hour or Not Selection (Checkbox)
is_peak_hour = st.checkbox('Is it peak hour?', value=False)
is_peak_hour = 1 if is_peak_hour else 0

# Road Surface Condition Selection (Radio button)
road_surface = st.radio(
    "Select the road surface condition:",
    ["Dry", "Wet"],
    index=0  # Default to Dry
)
road_surface_binary = 1 if road_surface == "Wet" else 0

# Weather Conditions (Radio button for each)
st.write("Select the weather condition:")
weather_condition = st.radio(
    "Weather condition:",
    ["Clear", "Foggy", "Rainy", "Snowy"],
    index=0  # Default to Clear
)

# Set the weather condition binary values
weather_condition_grouped_Foggy = 1 if weather_condition == "Foggy" else 0
weather_condition_grouped_Rainy = 1 if weather_condition == "Rainy" else 0
weather_condition_grouped_Snowy = 1 if weather_condition == "Snowy" else 0
weather_condition_grouped_Other = 0  # "Clear" is always set to 0

# Light Condition Selection (Checkbox)
light_condition = st.checkbox('Poor Visibility (Light Condition)', value=False)
light_condition_grouped_Poor_Visibility = 1 if light_condition else 0

# Speed Limit Selection (Radio button)
speed_limit = st.radio(
    "Select the speed limit group:",
    ["Low", "Medium", "High"],
    index=0  # Default to Low
)

group_speed_limit_Low = 1 if speed_limit == "Low" else 0
group_speed_limit_Medium = 1 if speed_limit == "Medium" else 0

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'urban_or_rural_area': [urban_or_rural_area],
    'is_weekend': [is_weekend],
    'is_peak_hour': [is_peak_hour],
    'road_surface_binary': [road_surface_binary],
    'weather_condition_grouped_Foggy': [weather_condition_grouped_Foggy],
    'weather_condition_grouped_Other': [weather_condition_grouped_Other],
    'weather_condition_grouped_Rainy': [weather_condition_grouped_Rainy],
    'weather_condition_grouped_Snowy': [weather_condition_grouped_Snowy],
    'light_condition_grouped_Poor Visibility': [light_condition_grouped_Poor_Visibility],
    'group_speed_limit_Low': [group_speed_limit_Low],
    'group_speed_limit_Medium': [group_speed_limit_Medium],
})

# Button to trigger the prediction
if st.button('Predict Accident Severity'):
    # Make prediction using the model
    prediction_proba = model.predict_proba(input_data)[:, 1]  # Probability for class 1 (severe)
    prediction = (prediction_proba >= best_threshold).astype(int)  # Apply the threshold to classify

    # Display result based on the prediction
    if prediction == 1:
        st.write("### Prediction Result:")
        st.write("There is a **high chance** for a **severe accident**.")
        st.write("""
            **Advice:** Given the high chance of a severe accident, it is strongly advised to consider alternative routes or delay your commute until conditions improve. 
            Stay updated on real-time traffic and weather reports before making any decisions.
        """)
    else:
        st.write("### Prediction Result:")
        st.write("There is a **low chance** for a **severe accident**.")
        st.write("""
            **Advice:** While the risk of a severe accident is low, remain cautious and continue to follow all safety guidelines when commuting.
            It's still a good practice to stay aware of current road and weather conditions.
        """)

