import streamlit as st
import joblib
import pandas as pd
import numpy as np
model = joblib.load('stellar_model_rfc.pkl')
# Title of the app
st.title('Stellar Object Classification')

# Input fields for each feature in your dataset
alpha = st.number_input("Right Ascension (alpha)", min_value=-180.0, max_value=180.0)
delta = st.number_input("Declination (delta)", min_value=-90.0, max_value=90.0)
u = st.number_input("Ultraviolet filter (u)")
g = st.number_input("Green filter (g)")
r = st.number_input("Red filter (r)")
i = st.number_input("Near Infrared filter (i)")
z = st.number_input("Infrared filter (z)")
redshift = st.number_input("Redshift value")


# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'alpha': [alpha],
    'delta': [delta],
    'u': [u],
    'g': [g],
    'r': [r],
    'i': [i],
    'z': [z],
    'redshift': [redshift],
})
# Make predictions when the user presses the button
if st.button('Classify Object'):
    prediction = model.predict(input_data)
    
    # Show the result
    st.write(f"The predicted class for the object is: {prediction[0]}")



