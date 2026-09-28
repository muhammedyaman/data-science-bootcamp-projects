import pandas as pd
import streamlit as st
from utils import load, slider

st.title("Online Payments Fraud Detection")
st.caption("Classification - Random Forest, no resampling (see the notebook for why SMOTE hurt precision here)")

bundle = load("fraud_detection")
r = bundle["ranges"]

transaction_type = st.selectbox("Transaction type", bundle["types"])
amount = slider(st, "Amount", r["amount"])
old_orig = slider(st, "Sender balance before", r["oldbalanceOrg"])
new_orig = slider(st, "Sender balance after", r["newbalanceOrig"])
old_dest = slider(st, "Receiver balance before", r["oldbalanceDest"])
new_dest = slider(st, "Receiver balance after", r["newbalanceDest"])

row = pd.get_dummies(pd.DataFrame([{"amount": amount, "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
                                    "oldbalanceDest": old_dest, "newbalanceDest": new_dest, "type": transaction_type}]),
                     columns=["type"]).reindex(columns=bundle["columns"], fill_value=False)

if st.button("Check for fraud"):
    proba = bundle["model"].predict_proba(row)[0, 1]
    st.metric("Estimated probability of fraud", f"{proba:.2%}")

st.divider()
st.caption(f"Test set: accuracy {bundle['metrics']['accuracy']:.4f}, F1 on the fraud class {bundle['metrics']['f1_fraud']:.3f}")
st.caption("The fraud share of the data is about 0.13%; accuracy alone is not informative here, see the notebook.")
