import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained objects
# -----------------------------
model = joblib.load("model/stress_rf_ordinal.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
label_encoder = joblib.load("model/ordinal_label_encoder.pkl")

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Student Stress Prediction", layout="centered")

st.title("🎓 Student Stress Level Prediction")
st.write(
    """
    This application predicts a student's **stress level** based on  
    their **study habits and learning environment**, using a trained  
    **Random Forest machine learning model**.
    """
)

st.subheader("📋 Study Habits Survey")

# -----------------------------
# User Inputs (Categorical)
# -----------------------------
study_resource = st.selectbox(
    "Primary study resource",
    [
        "Textbooks And Course",
        "Online Resources (eg Educational Websites, Ebooks)",
        "Video lectures or tutorials",
        "Audio recording and wildcard",
        "Library Resources"
    ]
)

study_location = st.selectbox(
    "Primary study location",
    [
        "Hostel Room",
        "School Library",
        "Study Hall Common Areas",
        "Cafes",
        "Outdoor (Parks, Campus Ground)"
    ]
)

device_used = st.selectbox(
    "Device used for studying",
    [
        "Laptop",
        "Desktop",
        "Tablets",
        "Smartphone",
        "Traditional Notes"
    ]
)

break_frequency = st.selectbox(
    "Break frequency during a typical study session",
    [
        "Every 30 minutes",
        "Every hour",
        "Every two hours",
        "I rarely take breaks during a study session"
    ]
)

# -----------------------------
# Build input feature vector
# -----------------------------
input_dict = {
    study_resource: 1,
    study_location: 1,
    device_used: 1,
    f"How Often Do You Take Breaks During A Typical Study Session_{break_frequency}": 1
}

input_data = pd.DataFrame([input_dict])

# Align with training features
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Stress Level"):
    prediction = model.predict(input_data)[0]
    stress_label = label_encoder.inverse_transform([prediction])[0]

    st.success(f"✅ Predicted Stress Level: **{stress_label}**")

    st.markdown(
        """
        **Note:**  
        This prediction is generated using a machine learning model trained on
        student stress survey data and should be interpreted as an indicative result.
        """
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("AI Coursework | Student Stress Prediction System")
