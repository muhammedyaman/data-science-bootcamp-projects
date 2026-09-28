import streamlit as st

st.set_page_config(page_title="Data Science Bootcamp Projects", page_icon="\U0001F4CA")

st.title("Data Science Bootcamp Projects")
st.markdown(
    """
This app serves the trained models of the projects in the
[GitHub repository](https://github.com) (see the sidebar for each project).

Every model here is retrained by `train_models.py` with the exact settings used in the
project notebook (same train/test split, same features), then saved with `joblib` and
loaded here for live predictions.

**Covered live in this app:** Regression (3), Classification (3), Clustering (3),
Recommendation Systems (3) and NLP (3) - 15 of the 30 projects, chosen because their
models are small and fast to serve.

**Not served live here:** Computer Vision, Time Series, Data Visualization, Deep Learning
and AI Agents. Their models are convolutional networks, LSTMs, YOLO weights or a local
Llama 3.2 model; they are too large or need hardware this free app does not have. See the
"Other Projects" page for their results and a link to the notebooks.
"""
)
