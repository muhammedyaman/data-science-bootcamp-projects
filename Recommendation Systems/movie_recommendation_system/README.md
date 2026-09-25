# Movie Recommendation System

Recommendation system project: recommending movies similar to a given movie from their content (plot, keywords, cast, director).

## Dataset

TMDB 5000 movies, from the links of the reference: [tmdb_5000_movies.csv](https://amanxai.com/wp-content/uploads/2020/05/tmdb_5000_movies.csv) (4,803 movies, 20 columns) and [tmdb_5000_credits.csv](https://amanxai.com/wp-content/uploads/2020/05/tmdb_5000_credits.csv) (cast and crew, 40 MB). Both files are in this folder.

## Reference solution

[Movie Recommendation System with Machine Learning](https://amanxai.com/2020/05/20/data-science-project-movie-recommendation-system/) (Aman Kharwal): TF-IDF vectors of the overviews (words and sequences of up to 3 words), a sigmoid kernel between all movies, and the ten most similar titles printed for a movie.

Issues found:
- No measure of the quality of the recommendations.
- The code fails for movies without an overview (`np.nan is an invalid document`), as several readers reported in the comments.
- Movies are looked up by `original_title`, which is not in English for 261 movies.
- The sigmoid kernel gives the same ranking as the cosine similarity (unit-length vectors, increasing function), so it only adds computation.
- The comments call the method "supervised", but a content-based recommender uses no labels.

## Our approach

1. Joined the two files, handled the missing overviews and used the English title.
2. Reproduced the reference recommendations.
3. Measured the quality with the genre overlap (Jaccard index) between a movie and its ten recommendations, averaged over all movies, against random and most-voted baselines. The genres are not an input of the compared methods.
4. Compared the reference with cosine similarity, single words, and a text that adds keywords, three actors and the director.

## Results

| Method | Genre overlap | Recommendations with a common genre |
|---|---|---|
| Overview, sigmoid kernel (reference) | 0.267 | 64.7% |
| Overview, cosine (identical ranking) | 0.267 | 64.7% |
| Overview + keywords + cast + director | 0.308 | 70.2% |
| Random | 0.168 | 49.5% |
| Ten most voted movies | 0.144 | 47.9% |

Content-based recommendations beat random ones, and the keywords, cast and director make them better. Genre overlap is only a proxy; real ratings would be needed to measure the true quality. See the notebook's Conclusion for details.
