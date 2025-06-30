import streamlit as st
import pandas as pd
import pickle

import sklearn
import sklearn.compose._column_transformer

# Manually add a dummy _RemainderColsList to avoid error
class _RemainderColsList(list):
    pass

sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

import pickle
with open('heart_model.pkl', 'rb') as f:
    model = pickle.load(f)


COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]


st.title("❤️ Heart Disease Prediction System")

st.markdown("""
This app predicts the **probability of Heart Disease** using your health parameters.
Fill in the details in the **sidebar**, then click **Predict**.
""")


# Sidebar - User Input
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 45)
    sex = st.sidebar.selectbox('Sex (0=Female, 1=Male)', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 600, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate', 70, 210, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.sidebar.slider('ST depression (oldpeak)', 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox('Slope of ST segment', [0, 1, 2])
    ca = st.sidebar.slider('Number of major vessels (ca)', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia (thal)', [0, 1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    return pd.DataFrame([data])

input_df = user_input_features()


st.subheader('✅ Your Input Parameters')
st.write(input_df)


if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('🎯 Prediction Result')
    if prediction[0] == 1:
        st.error('⚠️ Heart Disease Detected')
    else:
        st.success('✅ No Heart Disease Detected')

    st.subheader('📈 Prediction Probability')
    st.write(f"Probability of Heart Disease: **{prediction_proba[0][1]:.2f}**")
else:
    st.info('👈 Please enter parameters in the sidebar and click **Predict** to see results.')
