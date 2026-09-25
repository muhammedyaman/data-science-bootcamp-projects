# Sarcasm Detection

NLP project: classifying news headlines as sarcastic or not.

## Dataset

[Sarcasm.json](https://raw.githubusercontent.com/amankharwal/Website-data/master/Sarcasm.json): 26,709 headlines with the article link and a label `is_sarcastic`. The sarcastic headlines come from The Onion (satire), the others from HuffPost. Included in this folder (5.6 MB).

## Reference solution

[Sarcasm Detection with Machine Learning](https://amanxai.com/2021/08/24/sarcasm-detection-with-machine-learning/) (Aman Kharwal): `CountVectorizer` on the headlines, Bernoulli Naive Bayes, 80/20 split, accuracy 0.845, one headline tested by hand.

Issues found:
- The label is the source of the headline (The Onion or HuffPost, 100% in the data), so the model recognises a newspaper, not sarcasm; the article does not say so.
- 107 duplicated headlines are not removed.
- Only accuracy is reported and nothing shows what the model learnt.

## Our approach

1. Removed the duplicates and checked the label against the article link.
2. Reproduced the reference (accuracy 0.8448).
3. Compared Bernoulli and Multinomial Naive Bayes, Logistic Regression and Decision Tree on word counts and on words plus pairs of words, with precision, recall and F1 of the Onion class.
4. Read the words with the strongest weights and tested hand-written headlines.

## Results

| Model | Features | Accuracy | F1 (Onion) |
|---|---|---|---|
| Bernoulli Naive Bayes (reference) | words | 0.847 | 0.816 |
| Multinomial Naive Bayes | words and pairs | 0.853 | 0.833 |
| Logistic Regression | words and pairs | 0.847 | 0.826 |
| Decision Tree | words and pairs | 0.748 | 0.714 |

The strongest words are formulas of each newspaper ("area", "nation", "local" for The Onion), so the score measures how well two newspapers can be told apart. See the notebook's Conclusion for details.
