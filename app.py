import streamlit as st
import pandas as pd
import numpy as np
from preprocess import feature_engineering, load_model, num_cols, cat_cols

st.title("Income Prediction App (SVM)")

# Load model
model = load_model('model.pkl')

# Input form
def user_input_features():
    st.header("Enter Features")
    age = st.slider('Age', 17, 90, 30)
    workclass = st.selectbox('Workclass', [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
        'State-gov', 'Without-pay', 'Never-worked'
    ])
    fnlwgt = st.number_input('fnlwgt', 10000, 1000000, 200000)
    education = st.selectbox('Education', [
        'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm',
        'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate',
        '5th-6th', 'Preschool'
    ])
    education_num = st.slider('Education Num', 1, 16, 10)
    marital_status = st.selectbox('Marital Status', [
        'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-AF-spouse', 'Married-spouse-absent'
    ])
    occupation = st.selectbox('Occupation', [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
        'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ])
    relationship = st.selectbox('Relationship', [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
    ])
    race = st.selectbox('Race', [
        'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
    ])
    sex = st.selectbox('Sex', ['Female', 'Male'])
    capital_gain = st.number_input('Capital Gain', 0, 100000, 0)
    capital_loss = st.number_input('Capital Loss', 0, 5000, 0)
    hours_per_week = st.slider('Hours per Week', 1, 99, 40)
    native_country = st.selectbox('Native Country', [
        'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
        'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
        'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
        'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
        'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
        'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
        'Peru', 'Hong', 'Holand-Netherlands'
    ])
    data = {
        'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education': education,
        'education_num': education_num, 'marital_status': marital_status, 'occupation': occupation,
        'relationship': relationship, 'race': race, 'sex': sex, 'capital_gain': capital_gain,
        'capital_loss': capital_loss, 'hours_per_week': hours_per_week, 'native_country': native_country
    }
    return pd.DataFrame([data])

input_df = user_input_features()
input_df = feature_engineering(input_df)

if st.button('Predict'):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction")
    st.write(">50K" if pred == 1 else "<=50K")
    st.write(f"Probability of >50K: {proba:.2%}")