# Hate Speech Detection

NLP project: classifying tweets as hate speech, offensive language, or neither.

## Dataset

[twitter.csv](https://github.com/amankharwal/Website-data/raw/master/twitter.csv): 24,783 tweets, each with the number of annotators (3 or more), their votes for each class and the majority class (hate speech 5.8%, offensive language 77.4%, neither 16.8%). Included in this folder (2.5 MB).

## Reference solution

[Hate Speech Detection with Machine Learning](https://amanxai.com/2021/07/25/hate-speech-detection-with-machine-learning/) (Aman Kharwal): a cleaning function (lower case, punctuation, stop words, Snowball stemming), `CountVectorizer`, a Decision Tree on a 67/33 split, and one hand-written test sentence.

Issues found:
- No score by class: with 5.8% hate speech, the reproduced Decision Tree has an accuracy of 0.876 but a hate speech F1 of 0.35.
- 642 duplicated cleaned tweets (mostly retweets) are not removed; 33 of the 354 repeated texts have different labels in different copies.
- The test sentence has no insult words, and the model classifies it as "neither"; this is not discussed.

## Our approach

1. Cleaned the tweets with pandas string methods (retweet markers, mentions, links, HTML codes, non-letters) and removed empty and duplicated tweets (24,140 left).
2. Looked at the class balance and at the agreement between annotators.
3. Compared Decision Tree, Multinomial Naive Bayes and Logistic Regression (with and without `class_weight="balanced"`) on a stratified 80/20 split, with the scores of the hate speech class.
4. Read the words with the largest weights and tested hand-written sentences.

## Results

| Model | Accuracy | Macro F1 | Hate precision | Hate recall | Hate F1 |
|---|---|---|---|---|---|
| Decision Tree | 0.889 | 0.713 | 0.382 | 0.332 | 0.355 |
| Multinomial Naive Bayes | 0.885 | 0.651 | 0.406 | 0.148 | 0.217 |
| Logistic Regression | 0.903 | 0.720 | 0.475 | 0.274 | 0.348 |
| Logistic Regression, balanced | 0.868 | 0.735 | 0.312 | 0.603 | 0.411 |

The most accurate model finds only 27% of the hate speech tweets; class weights double the recall. The models learn slurs and profanity, not intent. See the notebook's Conclusion for details.
