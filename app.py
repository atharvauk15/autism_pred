import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Autism Prediction", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Try to load the model
try:
    model = load_model()
except:
    st.error("Failed to load model. Make sure 'best_model.pkl' exists in the repository.")
    model = None

# App title and description
st.title("Autism Spectrum Disorder Prediction")
st.write("""
This app predicts the likelihood of Autism Spectrum Disorder based on behavioral and personal features.
""")

# Create input form
st.header("Patient Information")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Add your input fields here based on the features in your model
    age = st.number_input("Age", min_value=1, max_value=100, value=28)
    gender = st.selectbox("Gender", ["Male", "Female"])
    jaundice = st.selectbox("Born with jaundice?", ["No", "Yes"])
    autism = st.selectbox("Family member with autism?", ["No", "Yes"])

with col2:
    # Add more input fields
    used_app = st.selectbox("Used autism screening app before?", ["No", "Yes"])
    result = st.number_input("Screening score", min_value=0, max_value=10, value=5)
    ethnicity = st.selectbox("Ethnicity", ["White European", "Hispanic", "Black", "Asian", "Middle Eastern", "Others"])
    country = st.selectbox("Country of residence", ["United States", "United Kingdom", "India", "Australia", "Other"])

# Add the A1-A10 questions (assuming these are the behavioral markers from your model)
st.header("Behavioral Markers")

col1, col2 = st.columns(2)

# Define the questions
questions = [
    "A1: I often notice small sounds when others do not",
    "A2: I usually concentrate more on the whole picture, rather than small details",
    "A3: I find it easy to do more than one thing at once",
    "A4: If there is an interruption, I can switch back very quickly",
    "A5: I find it easy to read between the lines",
    "A6: I know how to tell if someone listening to me is getting bored",
    "A7: When I'm reading a story, I find it difficult to work out the characters' intentions",
    "A8: I like to collect information about categories of things",
    "A9: I find it easy to work out what someone is thinking or feeling",
    "A10: I find it difficult to work out people's intentions"
]

# Create variables to store responses
responses = {}

# Create the questions split across two columns
with col1:
    for i in range(0, 5):
        responses[f'A{i+1}'] = 1 if st.selectbox(questions[i], ["No", "Yes"]) == "Yes" else 0

with col2:
    for i in range(5, 10):
        responses[f'A{i+1}'] = 1 if st.selectbox(questions[i], ["No", "Yes"]) == "Yes" else 0

# Prediction button
if st.button("Predict"):
    if model is not None:
        # Prepare input data (adjust according to your model's features)
        input_data = {
            'A1_Score': responses['A1'],
            'A2_Score': responses['A2'],
            'A3_Score': responses['A3'],
            'A4_Score': responses['A4'],
            'A5_Score': responses['A5'],
            'A6_Score': responses['A6'],
            'A7_Score': responses['A7'],
            'A8_Score': responses['A8'],
            'A9_Score': responses['A9'],
            'A10_Score': responses['A10'],
            'age': age,
            'gender': 1 if gender == "Male" else 0,
            'jaundice': 1 if jaundice == "Yes" else 0,
            'autism': 1 if autism == "Yes" else 0,
            'used_app_before': 1 if used_app == "Yes" else 0,
            'result': result,
            # You may need to encode ethnicity and country if your model expects that
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)
        pred_probability = model.predict_proba(input_df)
        
        # Display result
        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            st.warning("The model predicts: **Autism Spectrum Disorder detected**")
        else:
            st.success("The model predicts: **No Autism Spectrum Disorder detected**")
            
        st.write(f"Probability of ASD: {pred_probability[0][1]:.2%}")
        
        # Optional: Add a visualization
        st.subheader("Prediction Probability")
        fig, ax = plt.subplots()
        labels = ['No ASD', 'ASD']
        ax.bar(labels, [pred_probability[0][0], pred_probability[0][1]])
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        st.pyplot(fig)
