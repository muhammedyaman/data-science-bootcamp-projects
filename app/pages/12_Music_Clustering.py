import pandas as pd
import streamlit as st
from utils import load, slider

st.title("Clustering Music Genres")
st.caption("Clustering - K-Means (k=6) on 9 audio features (Spotify-2000); clusters follow the sound, not the labeled genre - see the notebook (NMI 0.07)")

bundle = load("music_clustering")
r = bundle["ranges"]

values = {}
cols = st.columns(3)
for i, name in enumerate(bundle["columns"]):
    values[name] = slider(cols[i % 3], name, r[name])

row = pd.DataFrame([values])[bundle["columns"]]
cluster = int(bundle["model"].predict(bundle["scaler"].transform(row))[0])
st.metric("Assigned cluster", f"Cluster {cluster}")

st.divider()
st.write("Cluster profiles (mean values):")
st.dataframe(bundle["profile"])
