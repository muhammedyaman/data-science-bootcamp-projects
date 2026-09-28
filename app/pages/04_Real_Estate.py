import pandas as pd
import streamlit as st
from utils import load, slider

st.title("Real Estate Price Prediction")
st.caption("Regression - Ridge Regression on the 2 features selected by p-value and correlation range")

bundle = load("real_estate")
r = bundle["ranges"]

distance = slider(st, "Distance to the nearest MRT station (meters)", r[bundle["columns"][0]])
stores = slider(st, "Number of convenience stores nearby", r[bundle["columns"][1]], step=1, fmt="%d")

row = pd.DataFrame([[distance, stores]], columns=bundle["columns"])
price = bundle["model"].predict(row)[0]
st.metric("Predicted price (per unit area)", f"{price:.1f}")

st.divider()
st.caption(f"Test set: R2 {bundle['metrics']['r2']:.3f}, MAE {bundle['metrics']['mae']:.2f}")
