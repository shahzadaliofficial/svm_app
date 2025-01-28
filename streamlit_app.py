import streamlit as st
import joblib

# Load the trained SVM model
def load_model(model_path):
    """Load the pre-trained SVM model."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Prediction function
def make_prediction(model, input_data):
    """Make predictions using the loaded model."""
    try:
        prediction = model.predict([input_data])[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Streamlit app
def main():
    # Page configuration
    st.set_page_config(
        page_title="SVM Classifier App",
        layout="centered",
        page_icon="🤖",
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

    # App title
    st.title("SVM Classifier Deployment 🤖")
    st.write("This app predicts whether a user will purchase a product based on their details.")

    # Load the model
    MODEL_PATH = "svm_model.pkl"  # Replace with your model path
    model = load_model(MODEL_PATH)

    if model is not None:
        # Input fields in the sidebar
        with st.sidebar:
            st.subheader("User Input Features")
            gender = st.radio("Gender", options=["Male", "Female"], index=0)
            age = st.slider("Age", min_value=18, max_value=70, value=30)
            salary = st.number_input("Estimated Salary (USD)", min_value=0, value=50000, step=1000)

            # Convert gender to numeric (Male: 1, Female: 0)
            gender_encoded = 1 if gender == "Male" else 0

            # Predict button
            if st.button("Predict Purchase"):
                # Prepare input data
                input_data = [gender_encoded, age, salary]

                # Make prediction
                prediction = make_prediction(model, input_data)

                # Display result
                if prediction is not None:
                    st.subheader("Prediction Result:")
                    if prediction == 1:
                        st.success("The user is likely to purchase the product. 🎉")
                    else:
                        st.error("The user is not likely to purchase the product. ❌")

# Run the app
if __name__ == "__main__":
    main()
