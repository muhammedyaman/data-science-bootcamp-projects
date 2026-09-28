import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from utils import load

st.title("Article Recommendation System")
st.caption("Recommendation - TF-IDF over title + text, cosine similarity")

bundle = load("article_recommendation")
titles = bundle["titles"]
title = st.selectbox("Article", sorted(titles))

index = titles.index(title)
scores = cosine_similarity(bundle["matrix"][index], bundle["matrix"])[0]
scores[index] = -1
top = scores.argsort()[::-1][:5]
st.write("Recommended articles:")
st.table({"article": [titles[i] for i in top], "similarity": [round(float(scores[i]), 3) for i in top]})
