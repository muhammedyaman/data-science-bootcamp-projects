# Mobile Price Classification

Classification project: predicting the price range (0 to 3) of a mobile phone from its specifications.

## Dataset

[mobile_prices.csv](https://raw.githubusercontent.com/amankharwal/Website-data/master/mobile_prices.csv), included in this folder. 2,000 phones, 20 features (battery power, RAM, screen size, camera pixels and others), 4 balanced price ranges of 500 phones each.

## Reference solution

[Mobile Price Classification with Machine Learning](https://amanxai.com/2021/03/05/mobile-price-classification-with-machine-learning/) (Aman Kharwal): `StandardScaler` on all data, 80/20 split, one Logistic Regression, accuracy 95.5%.

Issues found:
- Scaling before the split leaks test statistics into the training.
- Only one model and one accuracy number; no per-class metrics or confusion matrix.
- The printed count of predicted classes does not measure correctness.
- No check of impossible values (screen width of 0 in 180 phones).

## Our approach

1. Dropped `sc_w` (0 in 9% of phones, no correlation with the target) and 2 rows with a pixel height of 0.
2. Stratified 80/20 split with a training-only `StandardScaler`.
3. Compared six models on the test set and with a 5-fold cross-validation.
4. Analysed the errors per class and measured how much RAM alone explains.
5. Reproduced the reference (accuracy 0.955).

## Results

| Model | Test accuracy | CV accuracy |
|---|---|---|
| Logistic Regression | 0.958 | 0.964 +/- 0.006 |
| Gradient Boosting | 0.908 | 0.899 +/- 0.014 |
| Random Forest | 0.888 | 0.879 +/- 0.009 |
| Decision Tree | 0.800 | 0.829 +/- 0.008 |
| Naive Bayes | 0.812 | 0.811 +/- 0.020 |
| KNN | 0.568 | 0.524 +/- 0.020 |

Logistic Regression is the best model, and all its errors are between neighbouring price ranges. RAM alone gives 0.755 accuracy. See the notebook's Conclusion for details.
