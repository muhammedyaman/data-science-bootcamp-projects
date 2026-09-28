import pandas as pd
import streamlit as st
from utils import load, slider

st.title("Food Delivery Time Prediction")
st.caption("Regression - Gradient Boosting on courier age, rating, distance, vehicle and order type")

bundle = load("food_delivery")
r = bundle["ranges"]

age = slider(st, "Delivery person age", r["Delivery_person_Age"], step=1, fmt="%d")
rating = slider(st, "Delivery person rating", r["Delivery_person_Ratings"])
distance = slider(st, "Distance (km, haversine)", r["distance"])
vehicle = st.selectbox("Vehicle type", bundle["vehicles"])
order = st.selectbox("Order type", bundle["orders"])

row = pd.get_dummies(pd.DataFrame([{"Delivery_person_Age": age, "Delivery_person_Ratings": rating, "distance": distance,
                                    "Type_of_vehicle": vehicle, "Type_of_order": order}]),
                     columns=["Type_of_vehicle", "Type_of_order"]).reindex(columns=bundle["columns"], fill_value=False)

minutes = bundle["model"].predict(row)[0]
st.metric("Predicted delivery time", f"{minutes:.0f} minutes")

st.divider()
st.caption(f"Test set: R2 {bundle['metrics']['r2']:.3f}, MAE {bundle['metrics']['mae']:.1f} minutes")
