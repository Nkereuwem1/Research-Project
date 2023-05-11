# Import necessary libraries
import streamlit as st

# Define the app
def app():
    # Set the app title and description
    st.title('Drug Use Prediction App')
    st.write('Enter your information to check if you are likely to use cocaine, methamphetamine, heroin, or nicotine.')
    
    # Create input fields for user information
    age = st.slider('Age', 18, 65, 25)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    cocaine = st.selectbox('Cocaine Use', ['Yes', 'No'])
    meth = st.selectbox('Methamphetamine Use', ['Yes', 'No'])
    heroin = st.selectbox('Heroin Use', ['Yes', 'No'])
    nicotine = st.selectbox('Nicotine Use', ['Yes', 'No'])
    
    # Create a prediction button
    if st.button('Check'):
        # Check the user input and make a prediction
        if (age >= 25 and (gender == 'Male' or cocaine == 'Yes')) or (heroin == 'Yes' and meth == 'Yes') or nicotine == 'Yes':
            st.write('You are likely to use cocaine, methamphetamine, heroin, or nicotine.')
        else:
            st.write('You are unlikely to use cocaine, methamphetamine, heroin, or nicotine.')

# Run the app
if __name__ == '__main__':
    app()
