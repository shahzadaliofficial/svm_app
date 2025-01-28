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

# Minimal CSS Style
MINIMAL_STYLE = """
    <style>
    /* Main app background (adapts to dark/light mode) */
    .stApp {
        font-family: 'Arial', sans-serif;
    }

    /* Title styling */
    .title {
        font-size: 36px;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Button styling */
    .stButton>button {
        font-size: 16px;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        opacity: 0.8;
    }

    /* Input field styling */
    .stNumberInput input, .stRadio>div, .stSelectbox select {
        font-size: 14px;
        border-radius: 5px;
        padding: 10px;
    }

    /* Success message styling */
    .stSuccess {
        border-radius: 5px;
        padding: 10px;
    }

    /* Error message styling */
    .stError {
        border-radius: 5px;
        padding: 10px;
    }

    /* Form container styling */
    .stForm {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
"""

# Apply the minimal style
st.markdown(MINIMAL_STYLE, unsafe_allow_html=True)

# App title
st.markdown("<div class='title'>SVM Prediction App 🤖</div>", unsafe_allow_html=True)

# Load the model
MODEL_PATH = "svm_model.pkl"
model = load_model(MODEL_PATH)

if model is not None:
    # Input form
    with st.form("prediction_form"):
        st.subheader("Input User Data")
        gender = st.radio("Gender", ["Male", "Female"], index=0, help="Select the gender.")
        age = st.number_input("Age", min_value=0, max_value=100, step=None, placeholder="Enter age")
        salary = st.number_input("Estimated Salary", min_value=0, step=None, placeholder="Enter estimated salary")

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
