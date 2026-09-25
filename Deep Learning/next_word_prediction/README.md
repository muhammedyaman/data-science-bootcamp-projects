# Next Word Prediction

Deep learning project: predicting the next word of a sentence from the five previous words, on the book "The Adventures of Sherlock Holmes".

## Dataset

[1661-0.txt](https://www.gutenberg.org/files/1661/1661-0.txt): the book from Project Gutenberg (about 106,000 words without the header and license, 7,917 different words). Included in this folder.

## Reference solution

[Next Word Prediction Model](https://amanxai.com/2020/07/20/next-word-prediction-model/) (Aman Kharwal): sequences of five words, one-hot vectors for every word, an LSTM with one output per word, trained for 2 epochs, training curves shown.

Issues found:
- The one-hot arrays need about 5 GB of memory (4.2 GB inputs and 0.8 GB targets) for 7,917 words.
- The Project Gutenberg license text is part of the corpus.
- The text says 20 epochs but the code trains 2; there is no accuracy on unseen text and no comparison with a simple model.
- The function that converts a sentence to numbers raises an error for any word that is not in the book, and the article defines two different versions of it (characters and words).

## Our approach

1. Removed the header and license, tokenized the words as in the reference, and split the book in story order (80% train, 10% validation, 10% test).
2. Built the vocabulary from the training words (words seen at least 3 times, 2,742 words; the rest is "unknown") and used an Embedding layer instead of one-hot vectors (not in the course; the standard way to feed words to a network).
3. Trained an LSTM with early stopping and compared it with the most frequent word, a bigram and a trigram counting model on the top-1 and top-3 accuracy, and the perplexity.

## Results

| Model | Top-1 accuracy | Top-3 accuracy |
|---|---|---|
| Most frequent word | 5.2% | 12.0% |
| Bigram | 14.4% | 26.4% |
| Trigram with back-off | 14.6% | 24.6% |
| LSTM | 15.0% | 26.9% |

The LSTM (perplexity 109 on the test text) is not clearly better than a bigram model: the differences are within the statistical error, because a single book is too small. See the notebook's Conclusion for details.
