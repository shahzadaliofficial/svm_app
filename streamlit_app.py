import streamlit as st
import joblib

# Load the trained model
model = joblib.load("svm_model.pkl")

# Prediction function
def make_prediction(data):
    """Uses the pre-trained model to make predictions."""
    return model.predict([data])[0]

# Streamlit app configuration
st.set_page_config(page_title="SVM Classifier App", layout="wide", page_icon="📊")

# Header
st.header("SVM Classifier for Purchase Prediction 📊")

# Input form
st.sidebar.subheader("Input Data")
gender = st.sidebar.radio("Select Gender:", options=["Male", "Female"], index=0)
age = st.sidebar.slider("Enter Age:", min_value=18, max_value=70, step=1, value=25)
salary = st.sidebar.number_input("Enter Estimated Salary:", min_value=0, step=1000, value=30000)

# Convert gender to numeric
gender_encoded = 1 if gender == "Male" else 0

# Predict button
if st.sidebar.button("Predict"):
    # Prepare input data
    input_features = [gender_encoded, age, salary]
    result = make_prediction(input_features)

    # Display result
    st.subheader("Prediction Result:")
    if result == 1:
        st.success("The user is likely to purchase the product.")
    else:
        st.error("The user is not likely to purchase the product.")