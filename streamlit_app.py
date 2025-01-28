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
    /* Main app background with gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2); /* Light gradient background */
        color: #333333; /* Dark text */
        font-family: 'Arial', sans-serif;
    }

    /* Title styling */
    .title {
        font-size: 42px;
        color: #2c3e50; /* Dark blue text */
        font-family: 'Arial', sans-serif;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1); /* Subtle text shadow */
    }

    /* Button styling */
    .stButton>button {
        background-color: #3498db; /* Blue */
        color: #ffffff; /* White text */
        font-size: 18px;
        border-radius: 25px; /* Fully rounded corners */
        padding: 12px 24px;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    .stButton>button:hover {
        background-color: #2980b9; /* Darker blue on hover */
        transform: translateY(-2px); /* Slight lift on hover */
    }

    /* Input field styling */
    .stNumberInput input, .stSelectbox select {
        background-color: #ffffff; /* White background */
        color: #333333; /* Dark text */
        border-radius: 10px; /* Rounded corners */
        border: 1px solid #cccccc; /* Light gray border */
        padding: 12px;
        font-size: 16px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #3498db; /* Blue border on focus */
        box-shadow: 0 0 8px rgba(52, 152, 219, 0.5); /* Glow effect on focus */
    }

    /* Success message styling */
    .stSuccess {
        background-color: #2ecc71; /* Green */
        color: #ffffff; /* White text */
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    /* Error message styling */
    .stError {
        background-color: #e74c3c; /* Red */
        color: #ffffff; /* White text */
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    /* Form container styling */
    .stForm {
        background-color: #ffffff; /* White background */
        padding: 25px;
        border-radius: 15px; /* Rounded corners */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        margin: 20px auto;
        max-width: 600px; /* Limit form width */
    }

    /* Subheader styling */
    .stSubheader {
        font-size: 24px;
        color: #2c3e50; /* Dark blue text */
        font-weight: bold;
        margin-bottom: 15px;
    }

    /* Placeholder text styling */
    input::placeholder {
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
