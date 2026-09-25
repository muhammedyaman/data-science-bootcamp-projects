# Daily Births Forecasting

Time series project: forecasting the number of daily female births in California in 1959.

## Dataset

[daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv): 365 daily values (mean 42, standard deviation 7.3), no missing values. Included in this folder.

## Reference solution

[Daily Births Forecasting with Machine Learning](https://amanxai.com/2020/08/27/daily-births-forecasting-with-machine-learning/) (Aman Kharwal): a Facebook Prophet model with yearly seasonality, `changepoint_range=0.9`, `changepoint_prior_scale=0.5` and multiplicative seasonality, fitted on the whole year; 50 days are forecast and the plots are shown.

Issues found:
- No held-out data: the forecast is never compared with real values.
- No comparison with a simple model.
- Yearly seasonality is enabled with a single year of data (Prophet itself warns that about 2 years are needed).
- The fit on the training year (RMSE 6.5, against a standard deviation of 7.3) looks fine and hides the problem.

## Our approach

1. Explored the series (trend, weekly pattern, autocorrelation) and decomposed it.
2. Reproduced the reference model.
3. Evaluated forecasts of 30 days with an expanding window (6 origins) for the mean of the past, the last value, the mean of the last 7 days, ARIMA and Prophet with the reference and the default settings.

## Results

| Method | MAE | RMSE |
|---|---|---|
| ARIMA(0, 1, 1) | 5.85 | 7.51 |
| Prophet, default settings | 5.86 | 7.52 |
| Mean of the last 7 days | 5.98 | 7.58 |
| Mean of the past | 5.85 | 7.62 |
| Last value | 9.11 | 10.52 |
| Prophet, reference settings | 35.96 | 42.81 |

The series is mostly noise (the weekly pattern explains 4.7% of the variance), so a constant forecast at the level of the past mean is as good as ARIMA and Prophet with default settings. The reference settings give forecasts that fall towards zero. See the notebook's Conclusion for details.
