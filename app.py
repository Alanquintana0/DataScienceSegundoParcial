import streamlit as st
import pandas as pd
import numpy as np
import prediction
import joblib
from combined_attributes_adder import CombinedAttributesAdder


st.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg",width =50,use_column_width=3)
st.header('House Prediction Data Science proyect')
st.write('Data Science for predictions house value data')

col1, col2, col3, col4, col5 = st.columns(5)

with st.container():
    age = col1.number_input('Age', min_value=18, max_value=90, format="%d")
    cholesterol = col1.number_input('Cholesterol', min_value=120, max_value=400, format="%d")
    heart_rate = col1.number_input('Heart Rate', min_value=40, max_value=110, format="%d")
    diabetes = col1.number_input('Diabetes', min_value=0, max_value=1, format="%d")

with st.container():
    family_history = col2.number_input('Family History', min_value=0, max_value=1, format="%d")
    smoking = col2.number_input('Smoking', min_value=0, max_value=1, format="%d")
    obesity = col2.number_input('Obesity', min_value=0, max_value=1, format="%d")
    alcohol_consumption = col2.number_input('Alcohol Consumption', min_value=0, max_value=1, format="%d")

with st.container():
    exercise_hours_per_week = col3.number_input('Exercise Hours Per Week', min_value=0.0024423483189783, max_value=19.998709051535457, format="%.2f")
    previous_heart_problems = col3.number_input('Previous Heart Problems', min_value=0, max_value=1, format="%d")
    medication_use = col3.number_input('Medication Use', min_value=0, max_value=1, format="%d")
    stress_level = col3.number_input('Stress Level', min_value=1, max_value=10, format="%d")

with st.container():
    sedentary_hours_per_day = col4.number_input('Sedentary Hours Per Day', min_value=0.0012632057782457, max_value=11.999313410370352, format="%.2f")
    income = col4.number_input('Income', min_value=20062, max_value=299909, format="%d")
    bmi = col4.number_input('BMI', min_value=18.002336577801902, max_value=39.99721081557256, format="%.2f")
    triglycerides = col4.number_input('Triglycerides', min_value=30, max_value=800, format="%d")
    diet = col4.selectbox('Dieta', ['Unhealthy', 'Healthy'])

with st.container():
    physical_activity_days_per_week = col5.number_input('Physical Activity Days Per Week', min_value=0, max_value=7, format="%d")
    sleep_hours_per_day = col5.number_input('Sleep Hours Per Day', min_value=4, max_value=10, format="%d")
with st.container():
    model = col3.radio(
        'Select the model to use:',
        (
            'Random Forest Classificator',
        )
    )

    if st.button('Predict'):
        data = pd.DataFrame(
            {
                'Patient ID': ['UWX3861'],
                'Sex': ['Male'],
                'Blood Pressure': ['90/61'],
                'Diet': [diet]
                'Country': ['United Kingdom'],
                'Continent': ['Europe'],
                'Hemisphere': ['Northern Hemisphere'],
                'Age': [age],
                'Cholesterol': [cholesterol],
                'Heart_rate': [heart_rate],
                'Diabetes': [diabetes],
                'Family History': [family_history],
                'Smoking': [smoking],
                'Obesity': [obesity],
                'Alcohol Consumption': [alcohol_consumption],
                'Exercise Hours Per Week': [exercise_hours_per_week],
                'Previous Heart Problems': [previous_heart_problems],
                'Medication Use': [medication_use],
                'Stress Level': [stress_level],
                'Sedentary Hours Per Day': [sedentary_hours_per_day],
                'Income': [income],
                'BMI': [bmi],
                'Triglycerides': [triglycerides],
                'Physical Activity Days Per Week': [physical_activity_days_per_week],
                'Sleep Hours Per Day': [sleep_hours_per_day]
            }
        )

        result = prediction.predict(data, model)
        st.write("El riesgo de sufrir una enferedad cardiaca es: ".format(model,result[0]))
        