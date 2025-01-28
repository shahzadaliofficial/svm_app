import streamlit as st
import joblib

# Load the trained model
MODEL_PATH = "svm_model.pkl"
model = joblib.load(MODEL_PATH)

# Prediction function
def make_prediction(data):
    """Uses the pre-trained model to make predictions."""
    return model.predict([data])[0]

# Streamlit app configuration
st.set_page_config(
    page_title="SVM Classifier App",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #2E86C1;
    }
    .stSidebar {
        background-color: #f4f4f4;
        padding: 20px;
    }
    .stSuccess {
        color: #28a745;
    }
    .stError {
        color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.header("SVM Classifier for Purchase Prediction 📊")

# Input form in the sidebar
with st.sidebar:
    st.subheader("Input Data")
    gender = st.radio("Select Gender:", options=["Male", "Female"], index=0)
    age = st.slider("Enter Age:", min_value=18, max_value=70, step=1, value=25)
    salary = st.number_input("Enter Estimated Salary:", min_value=0, step=1000, value=30000)

    # Convert gender to numeric
    gender_encoded = 1 if gender == "Male" else 0

    # Predict button
    if st.button("Predict", key="predict_button"):
        # Prepare input data
        input_features = [gender_encoded, age, salary]
        result = make_prediction(input_features)

        # Display result
        st.subheader("Prediction Result:")
        if result == 1:
            st.success("The user is likely to purchase the product.")
        else:
            st.error("The user is not likely to purchase the product.")
