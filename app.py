import streamlit as st
import numpy as np
import joblib

st.title("Study Time Predictor")

model = joblib.load("study_model.pkl")

st.header("Enter Student Details")

gender = st.selectbox("Gender", ["Female", "Male"])
gender_val = 1 if gender == "Male" else 0

attendance_rate = st.slider("Attendance Rate", 0, 100, 75)

prev_grade = st.slider("Previous Grade", 0, 100, 70)

extra = st.selectbox("Extracurricular Activities", ["No", "Yes"])
extra_val = 1 if extra == "Yes" else 0

parental = st.selectbox("Parental Support", ["Low", "Medium", "High"])
parental_val = {"Low": 0, "Medium": 1, "High": 2}[parental]

attendance_percent = st.slider("Attendance (%)", 0, 100, 80)

online = st.selectbox("Online Classes Taken", ["No", "Yes"])
online_val = 1 if online == "Yes" else 0

input_data = np.array([[
    gender_val,
    attendance_rate,
    prev_grade,
    extra_val,
    parental_val,
    attendance_percent,
    online_val
]])

if st.button("Predict Study Hours"):
    prediction = model.predict(input_data)[0]
    st.success(f" Recommended Study Time: {prediction:.2f} hours/week")
