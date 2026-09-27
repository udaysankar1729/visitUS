import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")

if not os.path.exists(model_path):
    st.error("Model file not found! Please ensure 'best_model.joblib' is in the deployment folder.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

st.title("\u200B:airplane: Tourism Package Purchase Prediction")
st.markdown("\u200B:mag: Predict if a customer will purchase the Wellness Tourism Package")

# Input features from the user
st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age", 18, 80, 30)
type_of_contact = st.sidebar.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Government Sector"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.sidebar.slider("Number of Persons Visiting", 1, 10, 2)
preferred_property_star = st.sidebar.slider("Preferred Property Star Rating", 1, 5, 3)
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_of_trips = st.sidebar.slider("Number of Trips Annually", 0, 20, 5)
passport = st.sidebar.radio("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
own_car = st.sidebar.radio("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
num_children_visiting = st.sidebar.slider("Number of Children Visiting (below 5)", 0, 5, 0)
pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
num_followups = st.sidebar.slider("Number of Follow-ups", 0, 10, 3)
duration_of_pitch = st.sidebar.slider("Duration of Pitch (minutes)", 10, 60, 20)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'TypeofContact': [type_of_contact],
    'CityTier': [city_tier],
    'Occupation': [occupation],
    'Gender': [gender],
    'NumberOfPersonVisiting': [num_person_visiting],
    'PreferredPropertyStar': [preferred_property_star],
    'MaritalStatus': [marital_status],
    'NumberOfTrips': [num_of_trips],
    'Passport': [passport],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [num_children_visiting],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'NumberOfFollowups': [num_followups],
    'DurationOfPitch': [duration_of_pitch]
})

# Button to make prediction
if st.button("Predict Purchase"):
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f"The customer is LIKELY to purchase the package! (Probability: {prediction_proba[0]:.2f})")
        else:
            st.info(f"The customer is UNLIKELY to purchase the package. (Probability: {prediction_proba[0]:.2f})")

        st.markdown("--- Data Used for Prediction ---")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("**Note:** This is a predictive model. Results are based on the provided input and model training. Always use human judgment for final decisions.")
