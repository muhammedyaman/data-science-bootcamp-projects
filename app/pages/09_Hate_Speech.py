import re

import nltk
import streamlit as st
from utils import load

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

st.title("Hate Speech Detection")
st.caption("NLP - Logistic Regression with class_weight='balanced' (3 classes: hate speech, offensive language, neither)")

stemmer = nltk.SnowballStemmer("english")
stop_words = set(stopwords.words("english"))


def clean(text):
    text = text.lower()
    text = re.sub(r"rt @\w+:", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split(" ") if w not in stop_words]
    return " ".join(stemmer.stem(w) for w in words)


bundle = load("hate_speech")
tweet = st.text_area("Tweet", "the weather is so beautiful today, i love this city")

if tweet.strip():
    features = bundle["vectorizer"].transform([clean(tweet)])
    probabilities = dict(zip(bundle["model"].classes_, bundle["model"].predict_proba(features)[0]))
    st.bar_chart(probabilities)

st.divider()
st.caption(f"Test set: accuracy {bundle['metrics']['accuracy']:.3f}, macro F1 {bundle['metrics']['macro_f1']:.3f}")
st.caption("The label 'neither' is 3x more frequent than 'hate speech'; class_weight raises hate recall but lowers accuracy, see the notebook.")
