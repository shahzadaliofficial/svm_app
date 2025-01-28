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
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# New Modern CSS Style
MODERN_STYLE = """
    <style>
    /* Main app background */
    .stApp {
        background-color: #1e1e2f; /* Dark background */
        color: #ffffff; /* White text */
        font-family: 'Arial', sans-serif;
    }

    /* Title styling */
    .title {
        font-size: 36px;
        color: #ff6b6b; /* Vibrant red */
        font-family: 'Arial', sans-serif;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    /* Button styling */
    .stButton>button {
        background-color: #4ecdc4; /* Teal */
        color: #ffffff;
        font-size: 16px;
        border-radius: 25px; /* Rounded corners */
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #45b7af; /* Darker teal on hover */
    }

    /* Input field styling */
    .stNumberInput input, .stSelectbox select {
        background-color: #2d2d44; /* Dark input background */
        color: #ffffff; /* White text */
        border-radius: 10px; /* Rounded corners */
        border: 1px solid #4ecdc4; /* Teal border */
        padding: 10px;
        font-size: 14px;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #2d2d44; /* Dark sidebar background */
        padding: 20px;
        border-radius: 15px; /* Rounded corners */
    }

    /* Success message styling */
    .stSuccess {
        background-color: #4ecdc4; /* Teal */
        color: #ffffff;
        border-radius: 10px;
        padding: 10px;
    }

    /* Error message styling */
    .stError {
        background-color: #ff6b6b; /* Vibrant red */
        color: #ffffff;
        border-radius: 10px;
        padding: 10px;
    }

    /* Form container styling */
    .stForm {
        background-color: #2d2d44; /* Dark form background */
        padding: 20px;
        border-radius: 15px; /* Rounded corners */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
    }
    </style>
"""

# Apply the modern style
st.markdown(MODERN_STYLE, unsafe_allow_html=True)

# App title
st.markdown("<div class='title'>SVM Prediction App 🤖</div>", unsafe_allow_html=True)

# Load the model
MODEL_PATH = "svm_model.pkl"
model = load_model(MODEL_PATH)

if model is not None:
    # Input form
    with st.form("prediction_form"):
        st.subheader("Input User Data")
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
