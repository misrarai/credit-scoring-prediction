import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessor
model = joblib.load('best_credit_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# Define input fields
def user_input_features():
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    income = st.number_input('Annual Income', min_value=0, value=50000)
    home_ownership = st.selectbox('Home Ownership', ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    loan_amnt = st.number_input('Loan Amount', min_value=500, value=10000)
    loan_int_rate = st.number_input('Interest Rate (%)', min_value=0.0, max_value=100.0, value=10.0)
    loan_percent_income = st.number_input('Loan Percent Income', min_value=0.0, max_value=1.0, value=0.2)
    cb_person_default_on_file = st.selectbox('CB Person Default On File', ['Y', 'N'])
    cb_person_cred_hist_length = st.number_input('Credit History Length', min_value=0, value=5)
    person_emp_length = st.number_input('Employment Length (years)', min_value=0, value=5)
    data = {
        'person_age': age,
        'person_income': income,
        'person_home_ownership': home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length
    }
    return pd.DataFrame([data])

st.title('Credit Default Prediction')

input_df = user_input_features()

if st.button('Predict'):
    X = preprocessor.transform(input_df)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    st.write(f'Prediction: {"Default" if pred == 1 else "No Default"}')
    st.write(f'Probability of Default: {proba:.2f}')
