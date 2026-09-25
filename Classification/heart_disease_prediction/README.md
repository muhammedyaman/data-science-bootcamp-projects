# Heart Disease Prediction

Classification project: predicting heart disease from clinical measurements (Cleveland heart disease data, 303 patients).

## Dataset

[heart.csv](https://amanxai.com/wp-content/uploads/2020/05/heart.csv), included in this folder. 14 columns: age, sex, chest pain type (`cp`), resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate (`thalach`), exercise angina (`exang`), ST depression (`oldpeak`), ST slope, number of major vessels (`ca`), thalassemia (`thal`) and `target`.

## Reference solutions

Two articles by Aman Kharwal on the same data:
- [Heart Disease Prediction using Machine Learning](https://amanxai.com/2020/11/10/heart-disease-prediction-using-machine-learning/): Logistic Regression, 70/30 split, accuracy 86.8% on 91 test rows.
- [Heart Disease Prediction with Machine Learning](https://amanxai.com/2020/05/20/heart-disease-prediction-with-machine-learning/): KNN (k=12) and Random Forest with 10-fold cross-validation, accuracy only.

Issues found:
- The label is flipped in this version of the data: `target = 0` is the sick group (older, lower maximum heart rate, more angina and blocked vessels). The reference reads `target = 1` as disease and draws medically wrong conclusions (for example that no blocked vessels means higher risk).
- The data is called perfect, but it has 7 rows with impossible codes (`ca = 4`, `thal = 0`) and one duplicate.
- The scaler is fitted on all the data before cross-validation (leakage), and k is chosen on the same scores that are reported.
- Only accuracy is reported, on a test set of 91 patients. Recall of sick patients, the costly error in this setting, is not discussed.

## Our approach

1. Cleaned invalid codes and the duplicate; recoded the label after checking it against the clinical variables.
2. Stratified 80/20 split, one-hot encoding, `MinMaxScaler` fitted on the training set only.
3. Compared Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest and Gradient Boosting.
4. Validated with a leakage-free 10-fold cross-validation (scaler inside a pipeline) because a 60-patient test set is too noisy.
5. Reproduced the reference results to compare on equal terms.

## Results

10-fold cross-validation, 296 patients:

| Model | Accuracy | Recall (sick) | F1 |
|---|---|---|---|
| Logistic Regression | 0.834 +/- 0.050 | 0.807 +/- 0.104 | 0.815 +/- 0.060 |
| Naive Bayes | 0.841 +/- 0.078 | 0.777 +/- 0.147 | 0.811 +/- 0.115 |
| KNN (k=12) | 0.821 +/- 0.073 | 0.755 +/- 0.124 | 0.792 +/- 0.089 |
| Random Forest | 0.814 +/- 0.065 | 0.777 +/- 0.113 | 0.791 +/- 0.075 |
| Gradient Boosting | 0.790 +/- 0.076 | 0.732 +/- 0.102 | 0.762 +/- 0.080 |
| Decision Tree | 0.696 +/- 0.070 | 0.667 +/- 0.096 | 0.668 +/- 0.076 |

The first four models are within one standard deviation of each other. The scaler leak of the May article changes the KNN accuracy by only 0.003 here. See the notebook's Conclusion for details.
