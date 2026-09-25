# Food Delivery Time Prediction

Regression project: predicting food delivery time (minutes) from the delivery partner's age and rating, the distance between restaurant and delivery location, and the vehicle and order type.

## Dataset

`food_delivery.csv` (45,593 rows), the cleaned copy of the Kaggle [Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset) by Gaurav Malik, as linked in the reference article: https://raw.githubusercontent.com/ataislucky/Data-Science/main/dataset/food_delivery.txt

## Reference solution

"Food Delivery Time Prediction using Python" (Aman Kharwal). Only the copy published on Analytics Vidhya could be read, the original site was not reachable. It computes a haversine distance, then trains an LSTM on three features (age, rating, distance) and never evaluates the model on the test set.

Issues found:
- No test metric at all (no MAE, RMSE or R2).
- An LSTM on tabular data without any sequence structure, without feature scaling.
- Vehicle and order type are plotted but never used.
- The haversine code appeared to combine terms incorrectly (not fully verified).
- No data cleaning: rows with coordinates equal to 0 and ratings above 5 would distort the distance and rating features.

## Our approach

1. Cleaned invalid rows (4,071 with missing coordinates encoded as 0, 53 with rating 6.0).
2. Vectorized haversine distance, verified against a known case.
3. Compared Linear Regression, Ridge, Random Forest and Gradient Boosting on a held-out test set.
4. Checked what the reference skipped: vehicle type ablation, courier-level leakage, residuals, train versus test R2.

## Results

| Model | R2 | MAE (min) | RMSE (min) |
|---|---|---|---|
| Linear Regression | 0.296 | 6.23 | 7.85 |
| Ridge Regression | 0.296 | 6.23 | 7.85 |
| Random Forest | 0.318 | 6.04 | 7.73 |
| Gradient Boosting | 0.405 | 5.66 | 7.22 |

Predicting the mean gives an MAE of 7.56 minutes. R2 is low because the data lacks traffic, weather, preparation time and time of day, and the coordinates look synthetic. Train and test R2 are close, so the limit comes from the features, not the model. See the notebook's Conclusion for details.
