# Clustering Music Genres

Clustering project: grouping songs by their audio characteristics.

## Dataset

`Spotify-2000.csv` (1,994 popular songs, 15 columns: title, artist, top genre, year and audio features such as BPM, energy, danceability, loudness, valence, acousticness), the Kaggle [Spotify - All Time Top 2000s Mega Dataset](https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset), downloaded from a GitHub copy of the file because Kaggle requires a login.

## Reference solution

[Clustering Music Genres with Machine Learning](https://amanxai.com/2022/04/05/clustering-music-genres-with-machine-learning/) (Aman Kharwal): K-Means with k = 10 on six audio features and a 3D scatter plot.

Issues found:
- The scaling loop `for i in data.columns: MinMaxScaler(i)` never fits or applies a scaler, so tempo and length dominate the distances.
- The cluster names are mapped from 1 to 10, but K-Means labels clusters 0 to 9: the 175 songs of cluster 0 get no name.
- k = 10 is fixed without checking other values, three audio features are dropped without a reason, and the clusters are never compared with the genres, although the article is about music genres.

## Our approach

1. Converted `Length (Duration)` from text and standardized the nine audio features.
2. Chose k with `KElbowVisualizer` and the silhouette score; clustered with K-Means and drew the clusters in PCA space.
3. Profiled the clusters and compared them with the `Top Genre` column, which was not used for the clustering (cross-table and normalized mutual information).

## Results

Elbow: k = 6 (silhouette 0.185; the best silhouette is 0.208 for k = 2, so the groups overlap).

| Cluster | Songs | Main characteristic |
|---|---|---|
| Soft and acoustic | 517 | energy 33, acousticness 62 |
| Happy and danceable | 647 | danceability 66, valence 73 |
| Loud and dark | 550 | energy 73, loudness -7 dB, valence 38 |
| Live recordings | 128 | liveness 69 |
| Very long songs | 90 | about 9 minutes |
| Spoken / rap-like | 62 | speechiness 24 |

The clusters follow the sound, not the genre label (normalized mutual information with the genres 0.071). See the notebook's Conclusion for details.
