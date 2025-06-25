import pickle 
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Load your trained model
weather_model = pickle.load(open(r"weather_model.sav", 'rb'))

# Sidebar
with st.sidebar:st.markdown("<h2 style='text-align: center; color: #00aaff;'>The great weather prediction app of Ambalal Patel</h2>", unsafe_allow_html=True)


# Input layout
col1, col2 = st.columns(2)
with col1:
    temp = st.text_input("Enter temperature", value="32")
    humidity = st.text_input("Enter humidity", value="75")
    wind = st.text_input("Enter wind speed", value="10")
    precip = st.text_input("Enter precipitation", value="2")
    cloud = st.text_input("Enter cloud cover", value="50")
    atmosp = st.text_input("Enter atmospheric pressure", value="1012")
    
with col2:
    uv = st.text_input("Enter UV index", value="tarzan")
    season = st.text_input("Enter season (e.g. winter, summer)", value="0")
    visi = st.text_input("Enter visibility", value="10")
    location = st.text_input("Enter location", value="5")
    weatherty = st.text_input("Enter current weather type", value="8")

# Prediction button
if st.button("Predict"):

        # Convert numeric inputs
        input_data = [
            float(temp),
            float(humidity),
            float(wind),
            float(precip),
            float(cloud),
            float(atmosp),
            float(uv),
            float(season),
            float(visi),
            float(location)
        ]

        input_data_reshaped = np.array(input_data).reshape(1, -1)

        prediction = weather_model.predict(input_data_reshaped)

        if prediction[0] == 0:
            st.success("Rainy")
        elif prediction[0] == 1:
            st.success("Sunny")
        elif prediction[0] == 2:
            st.success("Overcast")
        elif prediction[0] == 3:
            st.success("Cloudy")
        elif prediction[0] == 4:
            st.success("Snowy")
    
