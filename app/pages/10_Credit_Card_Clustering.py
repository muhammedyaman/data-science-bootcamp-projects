import pandas as pd
import streamlit as st
from utils import load, slider

st.title("Credit Card Clustering")
st.caption("Clustering - K-Means (k=4) on 17 standardized features")

bundle = load("credit_card_clustering")
r = bundle["ranges"]
key_features = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "PURCHASES_FREQUENCY"]
st.caption("The 6 fields below drive the demo; the other 11 features are held at their median value.")

values = {c: r[c]["median"] for c in bundle["columns"]}
cols = st.columns(3)
for i, name in enumerate(key_features):
    values[name] = slider(cols[i % 3], name, r[name])

row = pd.DataFrame([values])[bundle["columns"]]
cluster = int(bundle["model"].predict(bundle["scaler"].transform(row))[0])
st.metric("Assigned segment", f"Cluster {cluster}")

st.divider()
st.write("Cluster profiles (mean values):")
st.dataframe(bundle["profile"])
