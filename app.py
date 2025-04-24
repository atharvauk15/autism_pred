import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Autism Prediction", layout="wide")

# Load the trained model and encoders
@st.cache_resource
def load_files():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    return model, encoders

# Try to load the model and encoders
try:
    model, encoders = load_files()
    st.sidebar.success("Model and encoders loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model or encoders: {e}")
    st.error("Make sure 'best_model.pkl' and 'encoders.pkl' exist in the repository.")
    model, encoders = None, None

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
    age = st.number_input("Age", min_value=1, max_value=100, value=28)
    gender = st.selectbox("Gender", ["Male", "Female"])
    jaundice = st.selectbox("Born with jaundice?", ["No", "Yes"])
    autism_family = st.selectbox("Family member with autism?", ["No", "Yes"])
    relation = st.selectbox("Who is completing the test", ["Self", "Parent", "Caregiver", "Medical staff", "Other"])

with col2:
    used_app = st.selectbox("Used autism screening app before?", ["No", "Yes"])
    ethnicity = st.selectbox("Ethnicity", ["White European", "Hispanic", "Black", "Asian", "Middle Eastern", "South Asian", "Turkish", "Others"])
    country = st.selectbox("Country of residence", ["United States", "United Kingdom", "India", "Australia", "New Zealand", "Others"])

# Add the A1-A10 questions (behavioral markers)
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

# Calculate screening score automatically
result = sum(responses.values())
st.info(f"Calculated screening score: {result}/10")

# Prediction button
if st.button("Predict"):
    if model is not None and encoders is not None:
        # Prepare input data - using EXACT column names from training
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
            'gender': gender,
            'jaundice': jaundice,
            'autism': autism_family,  # Using 'autism' to match training data
            'used_app_before': used_app,
            'result': result,
            'ethnicity': ethnicity,
            'country_of_res': country,  # Using 'country_of_res' to match training data
            'relation': relation
        }
    
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

       # Reorder columns to match training
        expected_order = [ 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                            'age', 'gender', 'ethnicity', 'jaundice', 'autism',
                            'country_of_res', 'used_app_before', 'result', 'relation']
        input_df = input_df[expected_order]

        
        # Apply the same encoders used during training
        for column in input_df.select_dtypes(include=['object']).columns:
            if column in encoders:
                try:
                    # Transform using the saved encoder
                    input_df[column] = encoders[column].transform([input_df[column].iloc[0]])[0]
                except ValueError:
                    # Handle unseen categories
                    st.warning(f"Warning: Unseen value in {column}. Using default encoding.")
                    input_df[column] = 0  # Assign a default value
        
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
        
        # Add a visualization
        st.subheader("Prediction Probability")
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ['No ASD', 'ASD']
        ax.bar(labels, [pred_probability[0][0], pred_probability[0][1]], color=['green', 'red'])
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        for i, v in enumerate([pred_probability[0][0], pred_probability[0][1]]):
            ax.text(i, v + 0.02, f"{v:.2%}", ha='center')
        st.pyplot(fig)
