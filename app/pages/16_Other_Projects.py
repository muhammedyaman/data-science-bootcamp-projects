import streamlit as st

st.title("Other Projects")
st.caption("Not served live in this app - see the reason for each category, and the notebook on GitHub for full results")

st.markdown(
    """
## Computer Vision
Flower Recognition, Colour Recognition, Count Objects in Image.
**Not served here:** the best models are a VGG16 transfer-learning network and a YOLOv8
detector; their weights are hundreds of megabytes and expect a GPU for reasonable speed.

## Time Series
Daily Births Forecasting, Website Traffic Forecasting, Time Series Forecasting with ARIMA.
**Not served here:** a forecast is a fixed answer for a fixed date range, not something a
single form input produces; the interesting part is the rolling-origin evaluation, which
is a notebook exercise, not a live query.

## Data Visualization
Uber Trips Analysis, Financial Budget Analysis, Billionaires Analysis.
**Not served here:** these projects are exploratory analyses of a fixed dataset (no model
to query with new input).

## Deep Learning
Stock Price Prediction with LSTM, Next Word Prediction, Image Classification with TensorFlow.
**Not served here:** the LSTM and CNN models are tens of megabytes each, need TensorFlow
at serving time and were trained for research comparisons (multiple seeds, baselines), not
for single-input queries.

## AI Agents
Chatbot Agent, Resume Screening Agent, Text Summarization Agent.
**Not served here:** these agents call a local Llama 3.2 model through Ollama, which this
online app cannot run. The notebooks include the results, with the language model
outputs cached in JSON files so they can be re-read without Ollama.

---
All 30 notebooks, with their saved outputs and a `README.md` per project, are in the
[GitHub repository](https://github.com).
"""
)
