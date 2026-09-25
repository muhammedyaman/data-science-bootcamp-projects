# Language Detection

NLP project: identifying the language of a text among 22 languages.

## Dataset

[dataset.csv](https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv): 22,000 Wikipedia sentences, 1,000 in each of 22 languages (Arabic, Chinese, Dutch, English, Estonian, French, Hindi, Indonesian, Japanese, Korean, Latin, Persian, Portuguese, Pushto, Romanian, Russian, Spanish, Swedish, Tamil, Thai, Turkish, Urdu). 141 rows are exact duplicates. Included in this folder (13 MB).

## Reference solution

[Language Detection with Machine Learning](https://amanxai.com/2021/10/30/language-detection-with-machine-learning/) (Aman Kharwal): word counts (`CountVectorizer`), Multinomial Naive Bayes, 67/33 split, one accuracy.

Issues found:
- Chinese and Japanese are written without spaces, so a sentence is a single "word" for a word counter; these languages are recognised much worse (recall 0.50 for Chinese in the reference split, 0.62 for Japanese in ours).
- The 141 duplicated sentences are not removed and the score is not looked at language by language.
- A reader asked what happens with a language that is not in the data; the article does not answer.

## Our approach

1. Removed the duplicates, fixed a misspelled label, and used a stratified 80/20 split (4,372 test sentences).
2. Compared Multinomial Naive Bayes on words (reference), Multinomial Naive Bayes" and Logistic Regression on character sequences of 1 to 3 characters, and the `langdetect` library.
3. Looked at the recall of every language, the confusion matrix, and tested five languages that are not in the data.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes, words (reference) | 0.956 | 0.957 |
| Naive Bayes, character sequences | 0.977 | 0.978 |
| Logistic Regression, character sequences | 0.984 | 0.984 |
| `langdetect` library | 0.877 | 0.818 |

Character sequences fix Chinese and Japanese (recall 0.99 and 0.985). Naive Bayes answers with a confidence of 1.0 even for unknown languages; Logistic Regression is less sure (0.26 to 0.37), which helps only partly. See the notebook's Conclusion for details.
