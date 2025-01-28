import streamlit as st
import joblib

# Load the pre-trained SVM model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Prediction function
def predict(model, input_data):
    try:
        prediction = model.predict([input_data])[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Streamlit app configuration
st.set_page_config(
    page_title="SVM Prediction App",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Modern CSS Style
MODERN_STYLE = """
    <style>
    /* Main app background */
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #333333; /* Dark text */
        font-family: 'Arial', sans-serif;
    }

    /* Title styling */
    .title {
        font-size: 42px;
        color: #4A90E2; /* Blue text */
        font-family: 'Arial', sans-serif;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4A90E2; /* Blue */
        color: #ffffff; /* White text */
        font-size: 18px;
        border-radius: 8px; /* Rounded corners */
        padding: 12px 24px;
        border: none;
        transition: background-color 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        background-color: #357ABD; /* Darker blue on hover */
    }

    /* Input field styling */
    .stNumberInput input, .stSelectbox select {
        background-color: #ffffff; /* White background */
        color: #333333; /* Dark text */
        border-radius: 8px; /* Rounded corners */
        border: 1px solid #cccccc; /* Light gray border */
        padding: 12px;
        font-size: 16px;
        width: 100%;
    }

    /* Success message styling */
    .stSuccess {
        background-color: #4CAF50; /* Green */
        color: #ffffff; /* White text */
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        text-align: center;
    }

    /* Error message styling */
    .stError {
        background-color: #ff4444; /* Red */
        color: #ffffff; /* White text */
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        text-align: center;
    }

    /* Form container styling */
    .stForm {
        background-color: #ffffff; /* White background */
        padding: 30px;
        border-radius: 15px; /* Rounded corners */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        margin: 0 auto;
        max-width: 600px;
    }

    /* Subheader styling */
    .stSubheader {
        font-size: 24px;
        color: #333333; /* Dark text */
        font-family: 'Arial', sans-serif;
        margin-bottom: 20px;
        font-weight: bold;
    }

    /* Placeholder text styling */
    ::placeholder {
        color: #999999; /* Light gray placeholder text */
    }
    </style>
"""

# Apply the modern style
st.markdown(MODERN_STYLE, unsafe_allow_html=True)

# App title
st.markdown("<div class='title'>SVM Prediction App</div>", unsafe_allow_html=True)

# Load the model
MODEL_PATH = "svm_model.pkl"
model = load_model(MODEL_PATH)

if model is not None:
    # Input form
    with st.form("prediction_form"):
        st.markdown("<div class='stSubheader'>Input User Data</div>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0, help="Select the gender.")
        age = st.number_input("Age", min_value=0, max_value=100, step=1, placeholder="Enter age")
        salary = st.number_input("Estimated Salary", min_value=0, step=1000, placeholder="Enter estimated salary")

        # Submit button
        submitted = st.form_submit_button("Predict")

        if submitted:
            # Preprocess input
            gender_numeric = 1 if gender == "Male" else 0
            user_input = [gender_numeric, age, salary]

            # Predict
            result = predict(model, user_input)

            # Display result
            if result is not None:
                st.markdown("### Prediction Result:")
                if result == 1:
                    st.success("The user is predicted to **Purchase** the product. 🎉")
                else:
                    st.error("The user is predicted to **Not Purchase** the product. ❌")
