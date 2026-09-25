# Credit Card Clustering

Clustering project: grouping credit card holders into behavioural segments.

## Dataset

`cc_general.csv` (8,950 customers, 6 months of credit card behaviour, 17 features), the Kaggle [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) (`CC GENERAL.csv`). The copy in this folder was downloaded from a public GitHub repository because Kaggle requires a login.

## Reference solution

[Credit Card Clustering with Machine Learning](https://amanxai.com/2022/10/03/credit-card-clustering-with-machine-learning/) (Aman Kharwal): K-Means with k = 5 on three features (`BALANCE`, `PURCHASES`, `CREDIT_LIMIT`) and a 3D scatter plot.

Issues found:
- The scaling loop `for i in columns: MinMaxScaler(i)` never fits or applies a scaler, so K-Means runs on raw values.
- `dropna()` removes 314 customers (3.5%) because of a column (`MINIMUM_PAYMENTS`) that is not used in the clustering.
- k = 5 is fixed without an elbow curve or silhouette score; no score is reported at all.
- Only 3 of the 17 features are used.

## Our approach

1. Filled the `MINIMUM_PAYMENTS` nulls with 0 after checking that they mean "no payment" (all 240 customers with no payment are null), and dropped the one row without a credit limit.
2. Showed on the reference's own features that scaling changes the clusters completely, and that both versions contain a tiny outlier cluster (24 to 29 customers).
3. Standardized all 16 behaviour columns and chose k with `KElbowVisualizer` and the silhouette score.
4. Compared K-Means, Agglomerative clustering and DBSCAN, and profiled the segments.

## Results

K-Means with k = 4 (elbow), silhouette 0.198 (agglomerative 0.176, DBSCAN finds no usable structure):

| Segment | Customers | Main behaviour |
|---|---|---|
| Big spenders | 409 | purchases 7,682, frequency 0.95, high limit |
| Regular buyers | 3,366 | purchases 1,236, frequency 0.89, few cash advances |
| Cash advance users | 1,197 | cash advances 4,522, high balance, 3% full payments |
| Low activity | 3,977 | purchases 270, frequency 0.17, low limit |

The low silhouette means the segments overlap; they describe behaviour rather than natural groups. See the notebook's Conclusion for details.
