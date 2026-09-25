# Online Payments Fraud Detection

Classification project: detecting fraudulent online payment transactions on the PaySim simulated mobile money dataset.

## Dataset

[PaySim on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) (`PS_20174392719_1491204439457_log.csv`, 6,362,620 rows, 8,213 frauds = 0.13%). The file is about 470 MB and is not included in this repository. Download it and save it as `dataset.csv` in this folder to run the notebook.

## Reference solution

[Online Payments Fraud Detection with Machine Learning](https://amanxai.com/2022/02/22/online-payments-fraud-detection-with-machine-learning/) (Aman Kharwal). It trains a Decision Tree on four features (`type` mapped to 1 to 5, `amount`, `oldbalanceOrg`, `newbalanceOrig`) with a 90/10 split and reports a single accuracy of 0.9997.

Issues found:
- Accuracy alone is meaningless with 0.13% fraud: always predicting "not fraud" gives 0.9987. No precision, recall, F1 or confusion matrix.
- Class imbalance is never mentioned.
- `type` is mapped to the numbers 1 to 5, which imposes a false order on the categories.
- Feature selection is not explained (correlations with the target are all near zero, yet the balance features are very informative).

## Our approach

1. Measured the imbalance and the always-"not fraud" baseline (accuracy 0.9987).
2. Dropped account identifiers, `isFlaggedFraud` and `step`; one-hot encoded `type`; stratified 80/20 split.
3. Tried undersampling + SMOTE on the training set (test set untouched) with four classifiers, and the same models without resampling.
4. Reproduced the reference model and compared feature sets with a 5-fold cross-validation.

## Results

Test set with 1,643 frauds among 1,272,524 transactions:

| Model | Precision | Recall | F1 | Missed frauds | False alarms |
|---|---|---|---|---|---|
| Decision Tree, no resampling | 0.908 | 0.900 | 0.904 | 164 | 150 |
| Random Forest, no resampling | 0.964 | 0.788 | 0.867 | 349 | 48 |
| Decision Tree, undersampling + SMOTE | 0.232 | 0.996 | 0.377 | 7 | 5,411 |
| Random Forest, undersampling + SMOTE | 0.240 | 0.996 | 0.387 | 7 | 5,172 |

- Resampling raised the recall but caused thousands of false alarms; the original data works better here.
- The reference model is reasonable (precision 0.90, recall 0.89) but this is invisible with accuracy alone. Our nine features give a small F1 gain in cross-validation (0.898 versus 0.882).
- `step` (time) must not be used: with time-ordered folds the F1 drops to 0.52 with a large variance, because the model memorizes when fraud happened.

See the notebook's Conclusion section for details and limitations.
