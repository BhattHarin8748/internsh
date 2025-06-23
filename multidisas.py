# -- coding: utf-8 --
"""
Created on Mon Jun 23 10:34:57 2025
@author: Yash
"""

import pickle 
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Load the saved models
diabetes_model = pickle.load(open("C:\\harin\\internship\\diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open("C:\\harin\\internship\\heart_disease_model.sav", 'rb'))
parkinson_model = pickle.load(open("C:\\harin\\internship\\parkinsons_data.sav", 'rb'))

# OPTIONAL: Load scalers if used during training
# Uncomment if you saved and used scalers
# heart_scaler = pickle.load(open("C:\\harin\\internship\\heart_scaler.sav", 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ------------------ DIABETES ------------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using SVM')

    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose Level')
        BloodPressure = st.text_input('Blood Pressure Value')
        SkinThickness = st.text_input('Skin Thickness Value')

    with col2:
        Insulin = st.text_input('Insulin Level')
        BMI = st.text_input('BMI Value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
        Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        try:
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            input_data_reshaped = np.array(input_data).reshape(1, -1)

            prediction = diabetes_model.predict(input_data_reshaped)

            if prediction[0] == 0:
                st.success('The person is not diabetic.')
            else:
                st.error('The person is diabetic.')
        except:
            st.warning("Please enter valid numerical values for all fields.")

# ------------------ HEART DISEASE ------------------
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using SVM')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Enter Age')
        sex = st.text_input('Enter Gender (0 = female, 1 = male)')
        cp = st.text_input('Chest Pain Type (0-3)')
        trestbps = st.text_input('Resting Blood Pressure')
        thal = st.text_input('Thal (0 = normal, 1 = fixed defect, 2 = reversible defect)')

    with col2:
        chol = st.text_input('Cholesterol Level')
        fbs = st.text_input('Fasting Blood Sugar (1 = true; 0 = false)')
        restecg = st.text_input('Resting ECG Results (0-2)')
        thalach = st.text_input('Maximum Heart Rate Achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
        oldpeak = st.text_input('ST Depression Induced by Exercise')
        slope = st.text_input('Slope of Peak Exercise ST Segment')
        ca = st.text_input('Number of Major Vessels (0-3)')

    if st.button('Heart Disease Test Result'):
        try:
            input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol),
                          float(fbs), float(restecg), float(thalach), float(exang),
                          float(oldpeak), float(slope), float(ca), float(thal)]

            input_data_reshaped = np.array(input_data).reshape(1, -1)

            # If you used a scaler during training:
            # input_data_reshaped = heart_scaler.transform(input_data_reshaped)

            prediction = heart_disease_model.predict(input_data_reshaped)

            if prediction[0] == 0:
                st.success('The person does NOT have a heart disease.')
            else:
                st.error('The person HAS a heart disease.')
        except:
            st.warning("Please enter valid numerical values for all fields.")

# ------------------ PARKINSON'S ------------------
if selected == 'Parkinsons Prediction':
    st.title('Parkinson’s Disease Prediction using SVM')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        fhi = st.text_input('MDVP:Fhi(Hz)')
        flo = st.text_input('MDVP:Flo(Hz)')
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col2:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        RAP = st.text_input('MDVP:RAP')
        PPQ = st.text_input('MDVP:PPQ')
        DDP = st.text_input('Jitter:DDP')

    with col3:
        Shimmer = st.text_input('MDVP:Shimmer')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        APQ3 = st.text_input('Shimmer:APQ3')
        HNR = st.text_input('HNR')

    with col4:
        APQ5 = st.text_input('Shimmer:APQ5')
        APQ = st.text_input('MDVP:APQ')
        DDA = st.text_input('Shimmer:DDA')
        NHR = st.text_input('NHR')

    if st.button("Parkinson's Test Result"):
        try:
            input_data = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                          float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                          float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR)]

            input_data_reshaped = np.array(input_data).reshape(1, -1)
            prediction = parkinson_model.predict(input_data_reshaped)

            if prediction[0] == 0:
                st.success("The person does NOT have Parkinson's disease.")
            else:
                st.error("The person HAS Parkinson's disease.")
        except:
            st.warning("Please enter valid numerical values for all fields.")
