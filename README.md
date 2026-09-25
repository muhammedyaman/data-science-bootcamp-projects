# Data Science Bootcamp Projects

Thirty projects in ten categories. Each project starts from a reference solution and its comments (articles by Aman Kharwal on [amanxai.com](https://amanxai.com) / thecleverprogrammer.com, from the list [230 Machine Learning Projects with Python](https://medium.com/coders-camp/230-machine-learning-projects-with-python-5d0c7abf8265)), checks it for methodological problems, and rebuilds it independently with a numerical evaluation.

## Method

For every project:

1. Read the reference solution and the reader comments.
2. Reproduce the reference result, so that any problem found is shown and not only claimed.
3. Rebuild the solution with a proper evaluation (train/test split or cross-validation, baselines, per-class metrics, residuals, ROUGE, and so on, depending on the task).
4. Compare with the reference and write down what differs and why.

Each project folder contains a notebook (with saved outputs), a short `README.md` (dataset, reference, issues found, results) and the data file, or a link to it when the file is too large for the repository.

## Projects

### Regression

| Project | Main finding |
|---|---|
| [Price Optimization](Regression/price_optimization) | The correlation between price and demand is confounded by differences between items; with item and store effects the price effect is close to zero. |
| [Real Estate Price Prediction](Regression/real_estate_price_prediction) | Feature selection by p-value and correlation range; Ridge Regression is the best model (test R2 0.554). |
| [Food Delivery Time Prediction](Regression/food_delivery_time_prediction) | Gradient Boosting reaches R2 0.405; the limit comes from the data, not from the model. |

### Classification

| Project | Main finding |
|---|---|
| [Online Payments Fraud Detection](Classification/online_payments_fraud_detection) | Accuracy 0.9997 is almost the same as always predicting "no fraud" (0.9987); precision and recall are needed. |
| [Heart Disease Prediction](Classification/heart_disease_prediction) | The target label of this dataset is inverted, and model differences are within noise on 296 patients. |
| [Mobile Price Classification](Classification/mobile_price_classification) | Logistic Regression is best (CV 0.964); KNN fails (0.52) because of many irrelevant features. |

### Clustering

| Project | Main finding |
|---|---|
| [Credit Card Clustering](Clustering/credit_card_clustering) | The scaling loop of the reference has no effect; four customer segments, but silhouette is only about 0.2. |
| [Customer Personality Analysis](Clustering/customer_personality_analysis) | Cluster numbers are arbitrary, so the segment names of the reference do not match; segments validated with campaign response. |
| [Clustering Music Genres](Clustering/clustering_music_genres) | Clusters follow the sound of the songs, not their genre (NMI 0.07). |

### Recommendation Systems

| Project | Main finding |
|---|---|
| [Movie Recommendation System](Recommendation%20Systems/movie_recommendation_system) | The sigmoid kernel and cosine similarity give identical rankings; adding keywords, cast and director improves genre overlap. |
| [Book Recommendation System](Recommendation%20Systems/book_recommendation_system) | The rating filter of the reference removes no rows; counting implicit interactions raises hit rate@10 from 10% to 16%. |
| [Article Recommendation System](Recommendation%20Systems/article_recommendation_system) | The reference call fails in current scikit-learn; duplicate articles recommend themselves. |

### NLP

| Project | Main finding |
|---|---|
| [Language Detection](NLP/language_detection) | Word counts fail for Chinese and Japanese; character n-grams fix it (accuracy 0.956 to 0.977). |
| [Sarcasm Detection](NLP/sarcasm_detection) | The label is the newspaper (The Onion vs HuffPost), so the models learn newspaper style, not sarcasm. |
| [Hate Speech Detection](NLP/hate_speech_detection) | Accuracy 0.876 hides an F1 of 0.35 for the hate class; class weights raise its recall from 0.27 to 0.60. |

### Computer Vision

| Project | Main finding |
|---|---|
| [Flower Recognition](Computer%20Vision/flower_recognition) | The reference uses the test set for validation; a small CNN overfits (0.999 train, 0.646 test), transfer learning with VGG16 reaches 0.828. |
| [Colour Recognition](Computer%20Vision/colour_recognition) | A nearest-neighbour classifier gives the same names as the interactive tool, much faster; RGB versus Lab distance cannot be ranked with this test. |
| [Count Objects in Image](Computer%20Vision/count_objects_in_image) | YOLOv8 counts are compared with counts by eye; the confidence threshold changes the result strongly. |

### Time Series

| Project | Main finding |
|---|---|
| [Daily Births Forecasting](Time%20Series/daily_births_forecasting) | The series is mostly noise; the mean forecast, ARIMA and default Prophet perform the same. |
| [Website Traffic Forecasting](Time%20Series/website_traffic_forecasting) | The real cycle is weekly, not 12 periods; a weekly SARIMA beats the reference (RMSE 915 against 1,179). |
| [Time Series Forecasting with ARIMA](Time%20Series/time_series_forecasting_arima) | The stock price behaves like a random walk; no ARIMA coefficient is significant. |

### Data Visualization

| Project | Main finding |
|---|---|
| [Uber Trips Analysis](Data%20Visualization/uber_trips_analysis) | The reference reads the weekday code wrongly; weekday totals must be divided by the number of days. |
| [Financial Budget Analysis](Data%20Visualization/financial_budget_analysis) | A typed total in the reference is wrong by 15,699.65 crores; two ministries hold 53.5% of the budget. |
| [Billionaires Analysis](Data%20Visualization/billionaires_analysis) | Dropping rows with missing age changes the country counts; the United States and China hold 49% of the billionaires. |

### Deep Learning

| Project | Main finding |
|---|---|
| [Stock Price Prediction with LSTM](Deep%20Learning/stock_price_prediction_lstm) | Repeating the last value (MAE 1.75) beats the LSTM on prices (MAE 4.09). |
| [Next Word Prediction](Deep%20Learning/next_word_prediction) | An embedding layer replaces one-hot vectors (5 GB); the LSTM is within noise of a bigram model. |
| [Image Classification with TensorFlow](Deep%20Learning/image_classification_tensorflow) | A CNN reaches 0.906 against 0.875 for the dense network of the reference; the shirt class is the weakest. |

### AI Agents

The agents are fixed workflows of tools (search, classifier, summarizer) with a local Llama 3.2 model run by [Ollama](https://ollama.com); no API key or online service is used. Ollama and the `llama3.2` model must be installed to rerun the language model cells; the resume screening and summarization notebooks cache the model outputs in JSON files, so they also run without it (the chatbot notebook calls the model live).

| Project | Main finding |
|---|---|
| [Chatbot Agent](AI%20Agents/chatbot_agent) | The reference network memorizes its 33 messages (0.55 on new ones); a confidence threshold sends unknown messages to the language model. |
| [Resume Screening Agent](AI%20Agents/resume_screening_agent) | 796 of 962 rows are copies; accuracy falls from 0.99 to 0.735 without them; the search tool reaches precision@5 of 0.79. |
| [Text Summarization Agent](AI%20Agents/text_summarization_agent) | TextRank is the best extractive method on ROUGE; the language model scores highest but adds words that are not in the text. |

## Running the notebooks

Python 3 with pandas, numpy, scikit-learn, matplotlib, seaborn and statsmodels covers most projects. Additional packages used by some projects: tensorflow (Keras), imbalanced-learn, yellowbrick, langdetect, prophet, yfinance, ultralytics, nltk, networkx, rouge-score and ollama.

Large data files are not stored in the repository. Each project README says where to download them (the fraud detection data, the Book-Crossing files, the TMDB credits, the Uber trips file and the flower photos), and the folder where to put them.
