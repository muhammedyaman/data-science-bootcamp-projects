import pandas as pd
import streamlit as st
from utils import load, slider

st.title("Mobile Price Classification")
st.caption("Classification - Logistic Regression, 4 price ranges (low, medium, high, very high)")

bundle = load("mobile_price")
binary_cols = [c for c, s in bundle["ranges"].items() if s["min"] == 0 and s["max"] == 1]
numeric_cols = [c for c in bundle["columns"] if c not in binary_cols]

values = {}
cols = st.columns(3)
for i, name in enumerate(numeric_cols):
    values[name] = slider(cols[i % 3], name, bundle["ranges"][name])
for i, name in enumerate(binary_cols):
    values[name] = int(cols[i % 3].checkbox(name))

row = pd.DataFrame([values])[bundle["columns"]]

if st.button("Predict"):
    pred = bundle["model"].predict(row)[0]
    st.metric("Predicted price range", ["Low cost", "Medium cost", "High cost", "Very high cost"][pred])

st.divider()
st.caption(f"Test set: accuracy {bundle['metrics']['accuracy']:.3f}, macro F1 {bundle['metrics']['macro_f1']:.3f}")
