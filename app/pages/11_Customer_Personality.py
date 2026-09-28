import pandas as pd
import streamlit as st
from utils import load, slider

st.title("Customer Personality Analysis")
st.caption("Clustering - K-Means (k=4) on income, seniority and spending; cluster numbers are arbitrary, so segments are read from the profile table")

bundle = load("customer_personality")
r = bundle["ranges"]

income = slider(st, "Income", r["Income"])
seniority = slider(st, "Seniority (months as a customer)", r["Seniority"])
spending = slider(st, "Total spending", r["Spending"])

row = pd.DataFrame([[income, seniority, spending]], columns=bundle["columns"])
cluster = int(bundle["model"].predict(bundle["scaler"].transform(row))[0])
st.metric("Assigned segment", f"Cluster {cluster}")

st.divider()
st.write("Cluster profiles (mean values and campaign response rate):")
st.dataframe(bundle["profile"])
