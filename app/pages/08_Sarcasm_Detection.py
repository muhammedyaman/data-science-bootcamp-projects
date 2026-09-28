import streamlit as st
from utils import load

st.title("Sarcasm Detection")
st.caption("NLP - Logistic Regression on words and word pairs (headline from The Onion vs HuffPost)")

bundle = load("sarcasm_detection")
headline = st.text_input("Headline", "area man wins argument with wife by staying silent for six hours")

if headline.strip():
    proba = bundle["model"].predict_proba(bundle["vectorizer"].transform([headline]))[0, 1]
    st.metric("Probability of being a sarcastic (Onion-style) headline", f"{proba:.0%}")

st.divider()
st.caption(f"Test set: accuracy {bundle['metrics']['accuracy']:.3f}, F1 {bundle['metrics']['f1']:.3f}")
st.caption("The label is really the newspaper, not sarcasm as a concept; see the notebook's Conclusion.")
