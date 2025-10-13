import streamlit as st
import joblib
import numpy as np
import os

# Set page config for a more professional look
st.set_page_config(page_title="Math Performance Predictor", layout="wide")

# Function to load model and scaler
@st.cache_resource
def load_assets():
    """Loads the pre-trained SVM model and scaler."""
    try:
        scaler_path = os.path.join('ASSETS', 'scaler.joblib')
        model_path = os.path.join('ASSETS', 'svm_best_model.joblib')
        scaler = joblib.load(scaler_path)
        svm_model = joblib.load(model_path)
        return scaler, svm_model
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Please make sure 'scaler.joblib' and 'svm_best_model.joblib' are in the 'ASSETS' directory.")
        return None, None

# Load assets
scaler, svm_model = load_assets()

# --- UI Layout ---
st.title("🧮 Math Performance Predictor")
st.markdown("Answer the following questions to predict a student's mathematical performance. Your honest responses will help in providing a more accurate prediction.")

# Define questions and options
questions = {
    "How confident are you in solving mathematical problems without help?": "Good",
    "How well do you understand basic mathematical concepts (fractions, algebra, geometry, etc.)?": "Excellent",
    "How regularly do you practice math problems apart from school homework?": "Average",
    "How easily can you apply formulas during exams or problem-solving?": "Good",
    "How well do you understand word problems (application-based questions)?": "Good",
    "How actively do you ask doubts or questions during math class?": "Average",
    "How accurate are you in your mathematical calculations during exams or assignments?": "Excellent",
    "How comfortable are you with logical reasoning and analytical thinking questions?": "Good",
    "How well do you remember mathematical formulas and theorems?": "Excellent",
    "How consistently do you score well in Mathematics exams?": "Good",
    "How much do you enjoy solving puzzles, riddles, or logic-based problems?": "Excellent",
    "How much time do you dedicate to studying Mathematics daily?": "Good",
    "How regularly do you revise previous math lessons before exams?": "Good",
    "How confident are you in explaining a math concept to your classmates?": "Excellent"
}

options = ["Very Poor", "Poor", "Average", "Good", "Excellent"]
value_mapping = {"Very Poor": 1, "Poor": 2, "Average": 3, "Good": 4, "Excellent": 5}

# Create two columns for inputs
col1, col2 = st.columns(2)
user_inputs = {}
question_keys = list(questions.keys())

# Display questions in two columns
for i, (question, default) in enumerate(questions.items()):
    default_index = options.index(default) if default in options else 2 # Default to 'Average' if not found
    if i < len(question_keys) / 2:
        with col1:
            user_inputs[question] = st.selectbox(question, options, index=default_index, key=f"q_{i}")
    else:
        with col2:
            user_inputs[question] = st.selectbox(question, options, index=default_index, key=f"q_{i}")

# --- Prediction Logic ---
st.write("") # Add some space
col1_button, col2_button, col3_button = st.columns([2.5,2,2])

with col2_button:
    predict_button = st.button("✨ Predict Performance", use_container_width=True)

if predict_button and scaler and svm_model:
    # Check for missing inputs (though selectbox always has a value)
    if any(val is None for val in user_inputs.values()):
        st.warning("Please answer all questions before predicting.")
    else:
        # Convert user selections to numeric values
        input_features = [value_mapping[val] for val in user_inputs.values()]
        input_array = np.array(input_features).reshape(1, -1)

        # Scale the input
        scaled_features = scaler.transform(input_array)

        # Predict using the SVM model
        prediction = svm_model.predict(scaled_features)
        predicted_class = prediction[0]

        # Mapping prediction to labels and reviews
        performance_labels = {
            1: "Very Low Performer",
            2: "Low Performer",
            3: "Average Performer",
            4: "High Performer",
            5: "Excellent Performer"
        }
        
        reviews = {
            "Very Low Performer": "Student needs significant improvement and consistent practice.",
            "Low Performer": "Student is below average; focus on understanding basic concepts and practice more.",
            "Average Performer": "Student is performing moderately; maintain regular practice to improve.",
            "High Performer": "Student is performing well; can aim for excellence with additional effort.",
            "Excellent Performer": "Student is excelling; continue current study habits and challenges."
        }

        predicted_label = performance_labels.get(predicted_class, "Unknown")
        review_comment = reviews.get(predicted_label, "No review available.")

        # Display the result in a visually appealing way
        st.write("---")
        st.subheader("Prediction Result")
        
        if predicted_label == "Very Low Performer" or predicted_label == "Low Performer":
            st.error(f"**Predicted Performance:** {predicted_label}")
        elif predicted_label == "Average Performer":
            st.warning(f"**Predicted Performance:** {predicted_label}")
        else:
            st.success(f"**Predicted Performance:** {predicted_label}")

        st.info(f"**Review & Suggestion:** {review_comment}")

# Add a footer
st.markdown("---")
st.markdown("Developed by an AI Assistant | Model: Support Vector Machine (SVM)")
