"""Retrains the final model of each project and saves it to app/models/.

Usage: python train_models.py [project_name ...]   (no argument: train everything)
The settings are the ones of the project notebooks (same splits and seeds).
"""
import sys
import warnings
from pathlib import Path

import json
import re
import string

import joblib
import nltk
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "models"
OUT.mkdir(exist_ok=True)


def save(name, **bundle):
    joblib.dump(bundle, OUT / f"{name}.joblib", compress=3)
    print("saved", name)


def ranges(df):
    return {c: {"min": float(df[c].min()), "max": float(df[c].max()), "median": float(df[c].median())} for c in df.columns}


def train_language_detection():
    data = pd.read_csv(ROOT / "NLP/language_detection/dataset.csv").drop_duplicates().reset_index(drop=True)
    data["language"] = data["language"].replace({"Portugese": "Portuguese"})
    x_train, x_test, y_train, y_test = train_test_split(data["Text"], data["language"], test_size=0.2, random_state=42, stratify=data["language"])
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3)).fit(x_train)
    models = {"Naive Bayes": MultinomialNB(), "Logistic Regression": LogisticRegression(max_iter=300)}
    metrics = {}
    for name, model in models.items():
        model.fit(vectorizer.transform(x_train), y_train)
        metrics[name] = {"accuracy": accuracy_score(y_test, model.predict(vectorizer.transform(x_test)))}
    save("language_detection", vectorizer=vectorizer, models=models, metrics=metrics)


def train_heart_disease():
    df = pd.read_csv(ROOT / "Classification/heart_disease_prediction/heart.csv").drop_duplicates()
    df = df[(df["ca"] < 4) & (df["thal"] > 0)].reset_index(drop=True)
    df["disease"] = 1 - df["target"]
    df = df.drop(columns="target")
    categorical = ["cp", "restecg", "slope", "ca", "thal"]
    x = pd.get_dummies(df.drop(columns="disease"), columns=categorical, drop_first=True)
    y = df["disease"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline(MinMaxScaler(), LogisticRegression()).fit(x_train, y_train)
    pred = model.predict(x_test)
    save("heart_disease", model=model, columns=list(x.columns), categorical=categorical,
         ranges=ranges(df[["age", "trestbps", "chol", "thalach", "oldpeak"]]),
         metrics={"accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred)})


def train_mobile_price():
    df = pd.read_csv(ROOT / "Classification/mobile_price_classification/mobile_prices.csv")
    df = df.drop(columns="sc_w")
    df = df[df["px_height"] > 0].reset_index(drop=True)
    x, y = df.drop(columns="price_range"), df["price_range"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x_train, y_train)
    pred = model.predict(x_test)
    save("mobile_price", model=model, columns=list(x.columns), ranges=ranges(x),
         metrics={"accuracy": accuracy_score(y_test, pred), "macro_f1": f1_score(y_test, pred, average="macro")})


def train_real_estate():
    df = pd.read_csv(ROOT / "Regression/real_estate_price_prediction/Real_Estate.csv")
    features = ["Distance to the nearest MRT station", "Number of convenience stores"]
    x, y = df[features], df["House price of unit area"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = Ridge().fit(x_train, y_train)
    pred = model.predict(x_test)
    save("real_estate", model=model, columns=features, ranges=ranges(x),
         metrics={"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred)})


def train_food_delivery():
    df = pd.read_csv(ROOT / "Regression/food_delivery_time_prediction/food_delivery.csv")
    for column in ["Type_of_order", "Type_of_vehicle"]:
        df[column] = df[column].str.strip()
    df = df[~((df["Restaurant_latitude"] <= 0) | (df["Restaurant_longitude"] <= 0)) & ~(df["Delivery_person_Ratings"] > 5)].reset_index(drop=True)
    lat1, lon1, lat2, lon2 = (np.radians(df[c]) for c in ["Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude"])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df["distance"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    features = ["Delivery_person_Age", "Delivery_person_Ratings", "distance", "Type_of_vehicle", "Type_of_order"]
    x = pd.get_dummies(df[features], drop_first=True)
    y = df["Time_taken(min)"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42).fit(x_train, y_train)
    pred = model.predict(x_test)
    save("food_delivery", model=model, columns=list(x.columns),
         ranges=ranges(df[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]]),
         vehicles=sorted(df["Type_of_vehicle"].unique()), orders=sorted(df["Type_of_order"].unique()),
         metrics={"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred)})


def train_price_optimization():
    df = pd.read_csv(ROOT / "Regression/price_optimization/Competition_Data.csv")
    keys = ["Store_ID", "Item_ID", "Fiscal_Week_ID"]
    agg = df.groupby(keys, as_index=False).agg(Price=("Price", "first"), Item_Quantity=("Item_Quantity", "first"))
    agg["log_Price"], agg["log_Item_Quantity"] = np.log(agg["Price"]), np.log(agg["Item_Quantity"])
    x = pd.concat([agg[["log_Price"]], pd.get_dummies(agg["Item_ID"], prefix="Item", drop_first=True),
                   pd.get_dummies(agg["Store_ID"], prefix="Store", drop_first=True)], axis=1)
    pooled = LinearRegression().fit(x, agg["log_Item_Quantity"])
    items = {}
    for item_id, group in agg.groupby("Item_ID"):
        if len(group) < 10 or group["Price"].std() == 0:
            continue
        model = LinearRegression().fit(group[["log_Price"]], group["log_Item_Quantity"])
        r = np.corrcoef(group["log_Price"], group["log_Item_Quantity"])[0, 1]
        items[str(item_id)] = {"model": model, "elasticity": float(model.coef_[0]), "r": float(r), "n": len(group),
                               "price_min": float(group["Price"].min()), "price_max": float(group["Price"].max()),
                               "current_price": float(group.sort_values("Fiscal_Week_ID")["Price"].iloc[-1])}
    save("price_optimization", pooled_model_elasticity=float(pooled.coef_[0]), items=items)


def train_fraud_detection():
    df = pd.read_csv(ROOT / "Classification/online_payments_fraud_detection/dataset.csv")
    x = pd.get_dummies(df.drop(columns=["isFraud", "nameOrig", "nameDest", "isFlaggedFraud", "step"]), columns=["type"], drop_first=True)
    y = df["isFraud"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1).fit(x_train, y_train)
    pred = model.predict(x_test)
    save("fraud_detection", model=model, columns=list(x.columns), types=sorted(df["type"].unique()),
         ranges=ranges(df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]]),
         metrics={"accuracy": accuracy_score(y_test, pred), "f1_fraud": f1_score(y_test, pred)})


def train_sarcasm_detection():
    data = pd.read_json(ROOT / "NLP/sarcasm_detection/Sarcasm.json", lines=True).drop_duplicates(subset="headline").reset_index(drop=True)
    x_train, x_test, y_train, y_test = train_test_split(data["headline"], data["is_sarcastic"], test_size=0.2, random_state=42, stratify=data["is_sarcastic"])
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2).fit(x_train)
    model = LogisticRegression(max_iter=1000).fit(vectorizer.transform(x_train), y_train)
    pred = model.predict(vectorizer.transform(x_test))
    save("sarcasm_detection", vectorizer=vectorizer, model=model,
         metrics={"accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred)})


def train_hate_speech():
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    stemmer = nltk.SnowballStemmer("english")
    stop_words = set(stopwords.words("english"))

    def clean(text):
        text = str(text).lower()
        text = re.sub(r"rt @\w+:", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"&\w+;", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = [w for w in text.split(" ") if w not in stop_words]
        return " ".join(stemmer.stem(w) for w in words)

    names = {0: "hate speech", 1: "offensive language", 2: "neither"}
    data = pd.read_csv(ROOT / "NLP/hate_speech_detection/twitter.csv", index_col=0)
    data["label"] = data["class"].map(names)
    data["clean"] = data["tweet"].apply(clean)
    data = data[data["clean"] != ""].drop_duplicates(subset="clean").reset_index(drop=True)

    x_train, x_test, y_train, y_test = train_test_split(data["clean"], data["label"], test_size=0.2, random_state=42, stratify=data["label"])
    vectorizer = CountVectorizer(stop_words="english", min_df=2).fit(x_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(vectorizer.transform(x_train), y_train)
    pred = model.predict(vectorizer.transform(x_test))
    save("hate_speech", vectorizer=vectorizer, model=model,
         metrics={"accuracy": accuracy_score(y_test, pred), "macro_f1": f1_score(y_test, pred, average="macro")})


def train_credit_card_clustering():
    df = pd.read_csv(ROOT / "Clustering/credit_card_clustering/cc_general.csv")
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(0)
    df = df.dropna().drop(columns="CUST_ID").reset_index(drop=True)
    scaler = StandardScaler().fit(df)
    x = scaler.transform(df)
    model = KMeans(4, n_init=10, random_state=42).fit(x)
    df["cluster"] = model.labels_
    profile = df.groupby("cluster")[["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "PURCHASES_FREQUENCY"]].mean().round(1)
    profile["customers"] = df["cluster"].value_counts().sort_index()
    save("credit_card_clustering", scaler=scaler, model=model, columns=list(df.columns[:-1]),
         ranges=ranges(df.drop(columns="cluster")), profile=profile)


def train_customer_personality():
    df = pd.read_csv(ROOT / "Clustering/customer_personality_analysis/marketing_campaign.csv", sep=";")
    df = df.dropna(subset=["Income"])
    df = df[(df["Income"] < 600000) & (df["Year_Birth"] >= 1920)].reset_index(drop=True)
    spending_columns = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Spending"] = df[spending_columns].sum(axis=1)
    df["Seniority"] = (pd.Timestamp("2014-10-04") - pd.to_datetime(df["Dt_Customer"])).dt.days / 30
    df["Age"] = 2014 - df["Year_Birth"]
    features = ["Income", "Seniority", "Spending"]
    scaler = StandardScaler().fit(df[features])
    x = scaler.transform(df[features])
    model = KMeans(4, n_init=10, random_state=42).fit(x)
    df["cluster"] = model.labels_
    profile = df.groupby("cluster")[features + ["Age"]].mean().round(0)
    profile["customers"] = df["cluster"].value_counts().sort_index()
    profile["campaign response rate"] = df.groupby("cluster")["Response"].mean().round(3)
    save("customer_personality", scaler=scaler, model=model, columns=features, ranges=ranges(df[features + ["Age"]]), profile=profile)


def train_music_clustering():
    df = pd.read_csv(ROOT / "Clustering/clustering_music_genres/Spotify-2000.csv")
    df["Length (Duration)"] = df["Length (Duration)"].str.replace(",", "").astype(int)
    audio = ["Beats Per Minute (BPM)", "Energy", "Danceability", "Loudness (dB)", "Liveness", "Valence", "Length (Duration)", "Acousticness", "Speechiness"]
    scaler = StandardScaler().fit(df[audio])
    x = scaler.transform(df[audio])
    model = KMeans(6, n_init=10, random_state=42).fit(x)
    df["cluster"] = model.labels_
    profile = df.groupby("cluster")[audio + ["Year", "Popularity"]].mean().round(1)
    profile["songs"] = df["cluster"].value_counts().sort_index()
    save("music_clustering", scaler=scaler, model=model, columns=audio, ranges=ranges(df[audio]), profile=profile)


def train_movie_recommendation():
    movies = pd.read_csv(ROOT / "Recommendation Systems/movie_recommendation_system/tmdb_5000_movies.csv")
    credits = pd.read_csv(ROOT / "Recommendation Systems/movie_recommendation_system/tmdb_5000_credits.csv")
    df = movies.merge(credits.rename(columns={"movie_id": "id"})[["id", "cast", "crew"]], on="id")
    df["overview"] = df["overview"].fillna("")
    df = df.drop_duplicates(subset="title").reset_index(drop=True)

    names = lambda text, n=None: [item["name"] for item in json.loads(text)][:n]
    df["keyword_list"] = df["keywords"].apply(names)
    df["cast_list"] = df["cast"].apply(lambda text: names(text, 3))
    df["director"] = df["crew"].apply(lambda text: [item["name"] for item in json.loads(text) if item["job"] == "Director"][:1])
    df["metadata"] = (df["overview"] + " " + df["keyword_list"].apply(lambda words: " ".join(w.replace(" ", "") for w in words))
                       + " " + df["cast_list"].apply(lambda words: " ".join(w.replace(" ", "") for w in words))
                       + " " + df["director"].apply(lambda words: " ".join(w.replace(" ", "") for w in words)))

    vectorizer = TfidfVectorizer(min_df=3, stop_words="english").fit(df["metadata"])
    matrix = vectorizer.transform(df["metadata"])
    save("movie_recommendation", vectorizer=vectorizer, matrix=matrix, titles=df["title"].tolist())


def train_book_recommendation():
    books = pd.read_csv(ROOT / "Recommendation Systems/book_recommendation_system/BX-Books.csv", sep=";", encoding="latin-1", on_bad_lines="skip")[["ISBN", "Book-Title"]]
    books.columns = ["isbn", "title"]
    users = pd.read_csv(ROOT / "Recommendation Systems/book_recommendation_system/BX-Users.csv", sep=";", encoding="latin-1", on_bad_lines="skip")
    users.columns = ["user", "location", "age"]
    ratings = pd.read_csv(ROOT / "Recommendation Systems/book_recommendation_system/BX-Book-Ratings.csv", sep=";", encoding="latin-1", on_bad_lines="skip")
    ratings.columns = ["user", "isbn", "rating"]

    user_counts = ratings["user"].value_counts()
    heavy = ratings[ratings["user"].isin(user_counts[user_counts >= 200].index)]
    titled = heavy.merge(books.drop_duplicates("isbn"), on="isbn")
    titled = titled[titled.groupby("title")["title"].transform("size") >= 50]
    titled = titled.merge(users, on="user")
    titled = titled[titled["location"].str.contains("usa|canada", na=False)].drop_duplicates(["user", "title"])

    table = titled.pivot(index="title", columns="user", values="rating").fillna(0)
    model = NearestNeighbors(metric="cosine", algorithm="brute").fit(csr_matrix(table.values))
    save("book_recommendation", model=model, table=csr_matrix(table.values), titles=table.index.tolist())


def train_article_recommendation():
    data = pd.read_csv(ROOT / "Recommendation Systems/article_recommendation_system/articles.csv", encoding="latin1")
    data = data.drop_duplicates(subset=["Article", "Title"]).reset_index(drop=True)
    data["Article"] = data["Article"].str.replace(r"[^\x00-\x7f]", " ", regex=True)
    combined = data["Title"] + ". " + data["Article"]
    vectorizer = TfidfVectorizer(stop_words="english").fit(combined)
    matrix = vectorizer.transform(combined)
    save("article_recommendation", vectorizer=vectorizer, matrix=matrix, titles=data["Title"].tolist())


TRAINERS = {name.removeprefix("train_"): fn for name, fn in globals().items()
            if name.startswith("train_") and getattr(fn, "__module__", None) == __name__}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(TRAINERS):
        TRAINERS[name]()
