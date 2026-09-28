import streamlit as st
from utils import load

st.title("Language Detection")
st.caption("NLP - character n-grams (1 to 3) + Naive Bayes / Logistic Regression")

bundle = load("language_detection")
text = st.text_area("Type a sentence", "Bugun hava cok guzel ve parkta yuruyus yapmaya gidiyoruz.")

if text.strip():
    features = bundle["vectorizer"].transform([text])
    cols = st.columns(2)
    for col, name in zip(cols, bundle["models"]):
        model = bundle["models"][name]
        pred = model.predict(features)[0]
        confidence = model.predict_proba(features).max()
        col.metric(name, pred, f"{confidence:.0%} confidence")

st.divider()
st.write("Test accuracy (held-out set):")
st.table({name: f"{m['accuracy']:.3f}" for name, m in bundle["metrics"].items()})
