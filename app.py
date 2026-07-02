"""
app.py — Heart Disease Prediction (Streamlit)

Trains the Logistic Regression model at startup directly from heart.csv
(instead of loading a pre-pickled model), which avoids scikit-learn
version-mismatch errors between your local machine and Streamlit Cloud.

Run locally:
    streamlit run app.py

Deploy:
    Push app.py, heart.csv, and requirements.txt to a GitHub repo,
    then deploy on https://share.streamlit.io (Streamlit Community Cloud).
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ---------------------------------------------------------------
# Train model (cached so it only runs once per app session)
# ---------------------------------------------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("heart.csv")
    X = df[FEATURE_ORDER]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

model, test_accuracy = load_and_train()

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter patient details below to estimate the likelihood of heart disease. "
    "This app uses a Logistic Regression model trained on the UCI Heart Disease dataset."
)
st.caption(f"Model test accuracy: {test_accuracy*100:.1f}%")
st.warning(
    "⚠️ This tool is for educational purposes only and is **not** a substitute "
    "for professional medical advice, diagnosis, or treatment."
)

st.divider()

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
    cp = st.selectbox(
        "Chest Pain Type", options=[
            ("Typical angina", 0), ("Atypical angina", 1),
            ("Non-anginal pain", 2), ("Asymptomatic", 3),
        ], format_func=lambda x: x[0]
    )
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0]
    )
    restecg = st.selectbox(
        "Resting ECG Result", options=[
            ("Normal", 0), ("ST-T wave abnormality", 1), ("Left ventricular hypertrophy", 2),
        ], format_func=lambda x: x[0]
    )

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exang = st.selectbox(
        "Exercise Induced Angina?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0]
    )
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", options=[
            ("Upsloping", 0), ("Flat", 1), ("Downsloping", 2),
        ], format_func=lambda x: x[0]
    )
    ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox(
        "Thalassemia", options=[
            ("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3),
        ], format_func=lambda x: x[0]
    )

st.divider()

# ---------------------------------------------------------------
# Predict
# ---------------------------------------------------------------
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([[
        age, sex[1], cp[1], trestbps, chol, fbs[1],
        restecg[1], thalach, exang[1], oldpeak, slope[1], ca, thal[1],
    ]], columns=FEATURE_ORDER)

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ **Heart Disease Detected** — estimated probability: {proba[1]*100:.1f}%")
    else:
        st.success(f"✅ **No Heart Disease Detected** — estimated probability of disease: {proba[1]*100:.1f}%")

    st.progress(float(proba[1]))

    with st.expander("See input summary"):
        st.dataframe(input_data)

st.divider()
st.caption("Model: Logistic Regression | Dataset: UCI Heart Disease")
