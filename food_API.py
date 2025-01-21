import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model (ensure it's saved first)
best_model = joblib.load('API_WEB_best_model.joblib')

# Define possible choices (you can replace these with your actual dataset values)
states = ['ADAMAWA', 'RIVERS', 'KANO', 'LAGOS', 'ABUJA']
zones = ['S.East', 'S.South', 'N.East', 'S.West', 'N.West']
desired_food_names = [
    'Beans brown, sold loose', 'Beans:white black eye. sold loose', 'Beef Bone in', 'Beef, boneless',
    'Broken Rice (Ofada)', 'Gari white, sold loose', 'Gari yellow, sold loose', 'Onion bulb',
    'Rice agric sold loose', 'Rice local sold loose', 'Rice Medium Grained', 
    'Rice, imported high quality sold loose', 'Tomato', 'Yam tuber', 'Irish potato', 'Sweet potato',
    'Palm oil: 1 bottle, specify bottle', 'Plantain(ripe)', 'Plantain(unripe)',
    'Wheat flour: prepacked (golden penny 2kg)'
]
years = list(range(2023, 2031))  # Years from 2023 to 2030
months = list(range(1, 13))  # Months from 1 to 12

# App title
st.title('Food Price Prediction App')

# Image container at the top
st.image('foodimage4.jpg', use_column_width=True)  # Replace with your image path or URL

# Quote container with custom size and color
quote_container = st.container()
with quote_container:
    st.markdown("""
        <p style="font-size: 20px; color: #efb7f7; font-style: italic;">
            "In agriculture, the price of food is as unpredictable as the weather, but it’s the farmers' resilience that sustains us through it all."
            - Dr Osifo O. Osifo 'Jan 2006
        </p>
    """, unsafe_allow_html=True)


# User input: Select state, zone, food_name, year, and month
state = st.selectbox('Select State', states)
zone = st.selectbox('Select Zone', zones)
food_name = st.selectbox('Select Food Item', desired_food_names)
year = st.slider('Select Year', 2023, 2030, 2023)  # Years range from 2023 to 2030
month = st.selectbox('Select Month', months)  # Months from 1 to 12

# Prepare the input data for prediction
user_input = pd.DataFrame({
    'state': [state],
    'zone': [zone],
    'foodName': [food_name],
    'year': [year],
    'month': [month]
})

# Display the input data
st.write('User Input Data:')
st.write(user_input)

# Fix for missing columns (ensure columns match the model's training data)
expected_columns = list(best_model.feature_names_in_)  # Use the feature names from the model

# One-hot encode the input
user_input_encoded = pd.get_dummies(user_input, columns=['state', 'zone', 'foodName'], drop_first=False)

# Ensure all required columns are present
for col in expected_columns:
    if col not in user_input_encoded.columns:
        user_input_encoded[col] = 0

# Reorder columns to match the model
user_input_encoded = user_input_encoded.reindex(columns=expected_columns, fill_value=0)

# Debugging: Display the encoded user input
st.write("Encoded User Input:")
st.write(user_input_encoded)

# Predict button
if st.button('Predict Price'):
    # Predict the price using the best model
    predicted_price = best_model.predict(user_input_encoded)

    # Display the predicted price
    st.write(f"The predicted price for {food_name} in {state} ({zone}) for {month}/{year} is: {predicted_price[0]:.2f}")
