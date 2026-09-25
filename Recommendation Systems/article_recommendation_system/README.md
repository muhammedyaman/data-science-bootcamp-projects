# Article Recommendation System

Recommendation system project: recommending blog articles similar to the article a reader is reading (content-based).

## Dataset

[articles.csv](https://raw.githubusercontent.com/amankharwal/Website-data/master/articles.csv): 34 excerpts (38 to 111 words) of articles of a data science blog, with their titles. One article is in the file twice. Included in this folder.

## Reference solution

[Article Recommendation System with Machine Learning](https://amanxai.com/2021/11/10/article-recommendation-system-with-machine-learning/) (Aman Kharwal): TF-IDF vectors of the texts, cosine similarity between all pairs, and the four recommended titles per article taken with `argsort()[-5:-1]`.

Issues found:
- `TfidfVectorizer(input=articles)` passes the texts as the `input` parameter, which raises an `InvalidParameterError` in the current scikit-learn.
- The trick `argsort()[-5:-1]` drops the article itself only if its similarity is unique: the duplicated article recommends itself.
- The four titles are listed from the least to the most similar.
- The result is checked by looking at one article; no measure of quality.

## Our approach

1. Reproduced the reference, showed the errors above, removed the duplicate and a non-readable character.
2. Assigned each article to a topic by hand from its title (9 topics, 31 articles in topics of at least two articles).
3. Compared the reference TF-IDF with TF-IDF on title and text, on words and pairs, and with plain word counts, against a random baseline, using the share of recommendations that have the same topic.

## Results

| Method | Top-1 in the same topic | Top-3 in the same topic (normalised) |
|---|---|---|
| TF-IDF, text (reference) | 77.4% | 70.4% |
| TF-IDF, title and text | 74.2% | 73.7% |
| TF-IDF, words and pairs, text | 74.2% | 71.5% |
| Word counts, text | 67.7% | 73.1% |
| Random | 3.2% | 10.8% |

Content-based recommendations are far better than random ones; with 31 evaluated articles the variants cannot be told apart. See the notebook's Conclusion for details and limitations.
