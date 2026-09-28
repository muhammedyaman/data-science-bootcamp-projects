import pandas as pd
import streamlit as st
from utils import load, slider

st.title("Heart Disease Prediction")
st.caption("Classification - Logistic Regression, scaler inside the pipeline")

bundle = load("heart_disease")
r = bundle["ranges"]

col1, col2 = st.columns(2)
with col1:
    age = slider(st, "Age", r["age"], step=1, fmt="%d")
    trestbps = slider(st, "Resting blood pressure", r["trestbps"], step=1, fmt="%d")
    chol = slider(st, "Cholesterol", r["chol"], step=1, fmt="%d")
    thalach = slider(st, "Max heart rate achieved", r["thalach"], step=1, fmt="%d")
    oldpeak = slider(st, "ST depression (oldpeak)", r["oldpeak"])
with col2:
    sex = st.radio("Sex", [1, 0], format_func=lambda v: "Male" if v else "Female")
    fbs = st.radio("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda v: "Yes" if v else "No")
    exang = st.radio("Exercise-induced angina", [0, 1], format_func=lambda v: "Yes" if v else "No")
    cp = st.selectbox("Chest pain type", [0, 1, 2, 3])
    restecg = st.selectbox("Resting ECG result", [0, 1, 2])
    slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia code (1-3)", [1, 2, 3])

row = pd.DataFrame([{"age": age, "sex": sex, "trestbps": trestbps, "chol": chol, "fbs": fbs, "restecg": restecg,
                     "thalach": thalach, "exang": exang, "oldpeak": oldpeak, "cp": cp, "slope": slope, "ca": ca, "thal": thal}])
x = pd.get_dummies(row, columns=["cp", "restecg", "slope", "ca", "thal"]).reindex(columns=bundle["columns"], fill_value=False)

if st.button("Predict"):
    proba = bundle["model"].predict_proba(x)[0, 1]
    st.metric("Probability of heart disease", f"{proba:.0%}")

st.divider()
st.caption(f"Test set: accuracy {bundle['metrics']['accuracy']:.3f}, F1 {bundle['metrics']['f1']:.3f}")
