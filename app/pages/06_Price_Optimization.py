import numpy as np
import pandas as pd
import streamlit as st
from utils import load

st.title("Price Optimization")
st.caption("Regression - per-item log-log elasticity (fixed effects), not the pooled correlation of the reference")

bundle = load("price_optimization")
items = bundle["items"]

st.write(f"Pooled elasticity (all items and stores together, confounded): {bundle['pooled_model_elasticity']:.3f}")

item_id = st.selectbox("Item", sorted(items, key=lambda k: items[k]["elasticity"]))
info = items[item_id]
st.metric("This item's elasticity (log-log slope)", f"{info['elasticity']:.2f}")
st.caption(f"correlation r = {info['r']:.2f} over {info['n']} observations, current price {info['current_price']:.2f}")

price = st.slider("Price to test", float(info["price_min"]), float(info["price_max"]), float(info["current_price"]))
predicted_log_quantity = info["model"].predict(pd.DataFrame({"log_Price": [np.log(price)]}))[0]
quantity = np.exp(predicted_log_quantity)
st.metric("Predicted quantity sold", f"{quantity:.1f}")
st.metric("Predicted revenue", f"{price * quantity:.1f}")

prices = np.linspace(info["price_min"], info["price_max"], 100)
predicted = np.exp(info["model"].predict(pd.DataFrame({"log_Price": np.log(prices)})))
st.line_chart(pd.DataFrame({"price": prices, "revenue": prices * predicted}).set_index("price"))
