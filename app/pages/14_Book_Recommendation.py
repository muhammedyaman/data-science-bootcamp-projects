import streamlit as st
from utils import load

st.title("Book Recommendation System")
st.caption("Recommendation - item-based k-NN (cosine) on ratings from heavy readers (>=200 ratings) in the US/Canada")

bundle = load("book_recommendation")
titles = bundle["titles"]
title = st.selectbox("Book", sorted(titles))

index = titles.index(title)
distances, neighbours = bundle["model"].kneighbors(bundle["table"][index], n_neighbors=6)
st.write("Recommended books:")
st.table({"book": [titles[i] for i in neighbours[0][1:]], "distance": [round(float(d), 3) for d in distances[0][1:]]})
