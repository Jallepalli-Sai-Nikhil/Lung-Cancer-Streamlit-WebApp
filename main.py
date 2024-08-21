import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as components


# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset for reference
dataset = pd.read_csv('dataset.csv')

# Define the label encoders manually
label_encoders = {
    'GENDER': LabelEncoder().fit(['Female', 'Male']),  # Female: 0, Male: 1 '
    'LUNG_CANCER': LabelEncoder().fit(['No', 'Yes'])  # No: 0, Yes: 1 # Use unique values from dataset
}

# Extract column names from dataset
feature_columns = dataset.drop('LUNG_CANCER', axis=1).columns.tolist()

def predict(input_data):
    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    
    # Ensure the DataFrame columns match those expected by the model
    input_df = input_df[feature_columns]
    
    # Predict using the trained model
    prediction = model.predict(input_df)
    
    # Decode the prediction
    return label_encoders['LUNG_CANCER'].inverse_transform(prediction)[0]



# Sidebar
st.sidebar.title('Navigation')
selection = st.sidebar.radio('Go to', ['Medical Information', 'Prediction'])

if selection == 'Prediction':
    st.subheader('Enter Patient Data')

    # Input form
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    smoking = st.selectbox('Smoking', [0, 1])
    yellow_fingers = st.selectbox('Yellow Fingers', [0, 1, 2])
    anxiety = st.selectbox('Anxiety', [0, 1, 2])
    peer_pressure = st.selectbox('Peer Pressure', [0, 1])
    chronic_disease = st.selectbox('Chronic Disease', [0, 1])
    fatigue = st.selectbox('Fatigue', [0, 1, 2])
    allergy = st.selectbox('Allergy', [0, 1])
    wheezing = st.selectbox('Wheezing', [0, 1, 2])
    alcohol_consuming = st.selectbox('Alcohol Consuming', [0, 1, 2])
    coughing = st.selectbox('Coughing', [0, 1, 2])
    shortness_of_breath = st.selectbox('Shortness of Breath', [0, 1, 2])
    swallowing_difficulty = st.selectbox('Swallowing Difficulty', [0, 1, 2])
    chest_pain = st.selectbox('Chest Pain', [0, 1, 2])

    # Create a dictionary for the input data
    input_data = dict(zip(feature_columns, [
        label_encoders['GENDER'].transform([gender])[0] if 'GENDER' in feature_columns else None,
        age,
        smoking,
        yellow_fingers,
        anxiety,
        peer_pressure,
        chronic_disease,
        fatigue,
        allergy,
        wheezing,
        alcohol_consuming,
        coughing,
        shortness_of_breath,
        swallowing_difficulty,
        chest_pain
    ]))

    # Predict button
    if st.button('Predict'):
        result = predict(input_data)
        st.write(f'Predicted Lung Cancer Status: {result}')
        
if selection == 'Medical Information':
    st.title('Lung Cancer Information')
    
    st.header('Nutrition and Lung Cancer Prevention')

    st.subheader('Foods Beneficial for Lung Cancer Prevention')

    st.markdown("""
    ### 1. Cruciferous Vegetables
    - **Examples**: Broccoli, cauliflower, cabbage, bok choy, brussels sprouts.
    - **Nutrients**: Contain sulforaphane, a compound believed to fight cancer, and indole-3-carbinol, which helps repair cell damage from carcinogens.

    ### 2. Orange Fruits and Vegetables
    - **Examples**: Oranges, tangerines, peaches, papayas, red bell peppers, carrots.
    - **Nutrients**: High in beta-cryptoxanthin, a carotenoid that may reduce the risk of lung cancer development and spread.

    ### 3. Leafy Green Vegetables
    - **Examples**: Spinach, kale, broccoli, romaine lettuce.
    - **Nutrients**: Rich in folate, which plays a key role in cellular repair and can protect against tobacco carcinogens, making it particularly beneficial for smokers.
    """)

    st.subheader('General Dietary Recommendations')
    st.markdown("""
    - Aim to fill at least half of your plate with fresh fruits and vegetables at every meal.
    - Maintain a diet low in red and processed meats and processed sugars to help manage body weight and blood sugar levels, both of which are important for cancer prevention.
    """)

    # General Information on Lung Cancer
    st.header('General Information on Lung Cancer')

    st.subheader('Symptoms of Lung Cancer')
    st.markdown("""
    - Persistent cough that worsens over time.
    - Coughing up blood or rust-colored sputum.
    - Chest pain that is often worse with deep breathing, coughing, or laughing.
    - Hoarseness.
    - Weight loss and loss of appetite.
    - Shortness of breath.
    - Fatigue or weakness.
    - Recurrent respiratory infections (like bronchitis or pneumonia).
    """)

    st.subheader('Types of Lung Cancer')
    st.markdown("""
    1. **Non-Small Cell Lung Cancer (NSCLC)**: 
    - The most common type, accounting for about 85% of lung cancer cases. It includes:
        - Adenocarcinoma
        - Squamous cell carcinoma
        - Large cell carcinoma

    2. **Small Cell Lung Cancer (SCLC)**:
    - Less common but more aggressive, often associated with smoking.
    """)

    st.subheader('Treatment Options for Lung Cancer')
    st.markdown("""
    - **Surgery**: Removal of the tumor and surrounding lung tissue.
    - **Radiation Therapy**: Using high-energy rays to kill cancer cells.
    - **Chemotherapy**: Using drugs to kill cancer cells or stop their growth.
    - **Targeted Therapy**: Drugs that target specific characteristics of cancer cells.
    - **Immunotherapy**: Treatments that help the immune system fight cancer.
    """)

    st.markdown("""
    For comprehensive and personalized information regarding lung cancer symptoms, types, and treatments, it is essential to consult healthcare professionals or oncologists.
    """)

    st.markdown("""
                # Doctors to visit:
                ### Oncologists
                """)
        
        
