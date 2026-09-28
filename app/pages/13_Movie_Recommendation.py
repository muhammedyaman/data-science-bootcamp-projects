import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from utils import load

st.title("Movie Recommendation System")
st.caption("Recommendation - TF-IDF over overview + keywords + cast + director, cosine similarity")

bundle = load("movie_recommendation")
titles = bundle["titles"]
title = st.selectbox("Movie", sorted(titles))

index = titles.index(title)
scores = cosine_similarity(bundle["matrix"][index], bundle["matrix"])[0]
scores[index] = -1
top = scores.argsort()[::-1][:10]
st.write("Recommended movies:")
st.table({"movie": [titles[i] for i in top], "similarity": [round(float(scores[i]), 3) for i in top]})
