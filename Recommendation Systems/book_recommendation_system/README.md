# Book Recommendation System

Recommendation system project: recommending books to readers from the ratings of all readers (item-based collaborative filtering).

## Dataset

The Book-Crossing dataset, from the links of the reference. The three files are not stored in this repository because of their size (about 120 MB); download them into this folder to run the notebook:
- [BX-Book-Ratings.csv](https://amanxai.com/wp-content/uploads/2020/05/BX-Book-Ratings.csv): 1,149,780 ratings (62% are zeros, meaning "read but not rated"; explicit ratings go from 1 to 10).
- [BX-Books.csv](https://amanxai.com/wp-content/uploads/2020/05/BX-Books.csv): 271,360 books.
- [BX-Users.csv](https://amanxai.com/wp-content/uploads/2020/05/BX-Users.csv): 278,858 readers with location and age.

## Reference solution

[Book Recommendation System with Machine Learning](https://amanxai.com/2020/05/23/book-recommendation-system-with-machine-learning/) (Aman Kharwal): readers with at least 200 ratings and books with at least 50 ratings, readers from the USA and Canada, a book-by-reader table filled with zeros, a cosine `NearestNeighbors` model, and the neighbours of a random book.

Issues found:
- The filter meant to remove books with less than 100 ratings is written on the values of the rating column, so it removes 0 rows.
- The zeros of the table mix "not rated" and "read but no rating given".
- No evaluation: nothing shows that the neighbours are books readers like.

## Our approach

1. Reproduced the reference filters and neighbours to show these points.
2. Hid 20% of the explicit ratings of readers with at least 5 ratings, and measured whether the models find, among unread books, the books the reader rated 8 or more (hit rate, recall and precision at 10, catalogue coverage), for 3,000 readers and a catalogue of 7,107 books.
3. Compared item-based cosine models built from the ratings, from the fact that a book was rated (binary), and from all interactions including zeros, with popularity and random baselines.

## Results

| Model | Hit rate@10 | Recall@10 | Precision@10 | Catalogue coverage |
|---|---|---|---|---|
| Random | 0.5% | 0.3% | 0.05% | 98.5% |
| Popularity | 5.9% | 2.8% | 0.65% | 0.65% |
| Item-based, ratings (reference idea) | 10.4% | 6.0% | 1.40% | 55.8% |
| Item-based, explicit ratings as binary | 10.2% | 6.0% | 1.39% | 54.6% |
| Item-based, all interactions as binary | 16.2% | 9.6% | 2.03% | 41.2% |

Counting the zeros as interactions helps the most; the rating values add nothing. See the notebook's Conclusion for details and limitations.
