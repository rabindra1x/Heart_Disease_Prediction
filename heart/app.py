import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('heart_model.pkl', 'rb'))

st.title("Heart Disease Prediction System")

st.markdown("""
This app predicts the **probability of Heart Disease** using your health parameters.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 45)
    sex = st.sidebar.selectbox('Sex', [0, 1]) # 0 = female, 1 = male
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol (chol)', 100, 600, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG (restecg)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate (thalach)', 70, 210, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.sidebar.slider('ST depression (oldpeak)', 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox('Slope of peak exercise ST segment', [0, 1, 2])
    ca = st.sidebar.slider('Number of major vessels (ca)', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia (thal)', [0, 1, 2, 3])

    data = [age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal]
    return np.array(data).reshape(1, -1)

input_data = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    st.subheader('Prediction Result')
    st.write('Heart Disease Detected' if prediction[0]==1 else 'No Heart Disease Detected')
    st.subheader('Prediction Probability')
    st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
