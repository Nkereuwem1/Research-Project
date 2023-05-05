# -*- coding: utf-8 -*-
"""
Created on Tue May  2 06:23:49 2023

@author: peezed
"""

# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd

# Load the trained models
with open(r"C:\Users\peezed\cocaine_model", "rb") as f:
    cocaine_model = pickle.load(f)
    
with open(r"C:\Users\peezed\meth_model", "rb") as f:
    meth_model = pickle.load(f)

with open(r"C:\Users\peezed\heroin_model", "rb") as f:
    heroin_model = pickle.load(f)


with open(r"C:\Users\peezed\nico_model", "rb") as f:
    nico_model = pickle.load(f)

# Define a function to preprocess user input
def preprocess_input(age, gender, education, heroin=None, *args):
    # Create a dictionary of variable names and input values
    input_dict = {'Age': [age], 'Gender': [gender], 'Education': [education]}

    # Add variables to the dictionary only if the user provided input for them
    variable_names = ['Amphet', 'Benzos', 'Cannabis', 'Coke', 'Crack', 'Ecstasy', 'Ketamine', 'Legalh', 'LSD', 'Mushrooms', 'VSA', 'Heroin', 'Nicotine']
    for i, arg in enumerate(args):
        if arg is not None:
            input_dict[variable_names[i]] = [arg]

    # Print the input dictionary for debugging
    print(input_dict)

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(input_dict)

    # Create binary indicators for drug use
    df['Cocaine_User'] = 0
    df.loc[df['Coke'] > 0, 'Cocaine_User'] = 1
    df = df.drop(['Coke'], axis=1)

    df['Meth_User'] = 0
    df.loc[df['Amphet'] > 0, 'Meth_User'] = 1
    df = df.drop(['Amphet'], axis=1)

    if 'Heroin' in df.columns:
        df['Heroin_User'] = 0
        df.loc[df['Heroin'] > 0, 'Heroin_User'] = 1
        df = df.drop(['Heroin'], axis=1)

    if 'Nicotine' in df.columns:
        df['Nicotine_User'] = 0
        df.loc[df['Nicotine'] > 0, 'Nicotine_User'] = 1
        df = df.drop(['Nicotine'], axis=1)

    # Drop the columns that were not used
    df = df.drop(['Benzos', 'Cannabis', 'Crack', 'Ecstasy', 'Ketamine', 'Legalh', 'LSD', 'Mushrooms', 'VSA'], axis=1)

    return df

# Create the web app
st.title("Stimulant Use Prediction")

age = st.number_input("Age", min_value=0, max_value=99, value=25, step=1)
gender = st.selectbox("Gender", options=['Male', 'Female'])
education = st.selectbox("Education", options=['Left school before 16 years', 'Left school at 16 years',
                                               'Left school at 17 years', 'Left school at 18 years',
                                               'Some college or university, no certificate or degree',
                                               'Professional certificate/ diploma', 'University degree',
                                               'Masters degree', 'Doctorate degree'])
country = st.selectbox('Country', ['Australia', 'Canada', 'New Zealand', 'Other', 'Republic of Ireland', 'UK', 'USA'])
ethnicity = st.selectbox('Ethnicity', ['Asian', 'Black', 'Mixed-Black/Asian', 'Mixed-White/Asian', 'Mixed-White/Black', 'Other', 'White'])
amphet = st.slider("Amphet Use (Past Year)", min_value=0, max_value=6, step=1)
benzos = st.slider("Benzos Use (Past Year)", min_value=0, max_value=6, step=1)
cannabis = st.slider("Cannabis Use (Past Year)", min_value=0, max_value=6, step=1)
coke = st.slider("Cocaine Use (Past Year)", min_value=0, max_value=6, step=1)
crack = st.slider("Crack Use (Past Year)", min_value=0, max_value=6, step=1)
ecstasy = st.slider("Ecstasy Use (Past Year)", min_value=0, max_value=6, step=1)
ketamine = st.slider("Ketamine Use (Past Year)", min_value=0, max_value=6, step=1)
legalh = st.slider("Legal Highs Use (Past Year)", min_value=0, max_value=6, step=1)
lsd = st.slider("LSD Use (Past Year)", min_value=0, max_value=6, step=1)
mushrooms = st.slider("Mushrooms Use (Past Year)", min_value=0, max_value=6, step=1)
vsa = st.slider("Volatile Substance Use (Past Year)", min_value=0, max_value=6, step=1)
nicotine = st.slider("Nicotine Use (Past Year)", min_value=0, max_value=6, step=1)

#Preprocess the user input
input_data = preprocess_input(age, gender, education, amphet, benzos, cannabis, coke, crack, ecstasy, ketamine, legalh, lsd, mushrooms, vsa, nicotine)

#Display the user input
st.subheader("Input:")
st.write(input_data)

# Make the predictions
if coke > 0:
    coke_prediction = cocaine_model.predict(input_data)
    if coke_prediction == 1:
        st.write("You are predicted to have cocaine use disorder.")
    else:
        st.write("You are not predicted to have cocaine use disorder.")
else:
    st.write("You did not report cocaine use in the past year.")

if input_data['Meth_User'][0] == 1:
    meth_prediction = meth_model.predict(input_data)
    if meth_prediction == 1:
        st.write("You are predicted to have methamphetamine use disorder.")
    else:
        st.write("You are not predicted to have methamphetamine use disorder.")
else:
    st.write("You did not report methamphetamine use in the past year.")

if input_data['Heroin_User'][0] == 1:
    heroin_prediction = heroin_model.predict(input_data)
    if heroin_prediction == 1:
        st.write("You are predicted to have heroin use disorder.")
    else:
        st.write("You are not predicted to have heroin use disorder.")
else:
    st.write("You did not report heroin use in the past year.")

if nicotine > 0:
    nico_prediction = nico_model.predict(input_data)
    if nico_prediction == 1:
        st.write("You are predicted to have nicotine use disorder.")
    else:
        st.write("You are not predicted to have nicotine use disorder.")
else:
    st.write("You did not report nicotine use in the past year.")
