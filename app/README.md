---
title: Data Science Bootcamp Projects
emoji: "\U0001F4CA"
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
---

# Data Science Bootcamp Projects - Streamlit Demo

Live demo of 15 of the 30 projects in the
[GitHub repository](https://github.com) (data science bootcamp: 10 categories, 3 projects each,
each one comparing a reference solution with an independent rebuild).

Each page loads a model retrained with the exact settings of its project notebook
(`train_models.py` in the repository) and lets you enter values and see a live prediction:
Regression, Classification, Clustering, Recommendation Systems and NLP. The other 5
categories (Computer Vision, Time Series, Data Visualization, Deep Learning, AI Agents) are
not served live here; see the "Other Projects" page in the app for why, and the notebooks
on GitHub for their full results.

## Running locally

```
pip install -r requirements.txt
python train_models.py       # retrains every model and writes app/models/*.joblib
streamlit run app.py
```

`train_models.py` reads the data files from the sibling project folders (for example
`../Classification/heart_disease_prediction/heart.csv`), so it must be run from inside
this `app/` folder with the rest of the repository checked out next to it.
