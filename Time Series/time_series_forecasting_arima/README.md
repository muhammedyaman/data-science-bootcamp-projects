# Time Series Forecasting with ARIMA

Time series project: forecasting the closing price of the Google stock with ARIMA models.

## Dataset

`google_stock.csv`: 252 trading days of Google (GOOG) from 21 June 2021 to 17 June 2022, downloaded with `yfinance` as in the reference. The prices are adjusted for the 20 to 1 share split of July 2022, so they are about 20 times smaller than the ones printed in the article. Included in this folder.

## Reference solution

[Time Series Forecasting with ARIMA](https://amanxai.com/2022/06/21/time-series-forecasting-with-arima/) (Aman Kharwal): a decomposition with a period of 30 days to decide that the prices are "seasonal", orders (5, 1, 2) read by eye from autocorrelation plots, an ARIMA model, then a SARIMA model with a seasonal period of 12 fitted on the whole year.

Issues found:
- The price is a random walk (returns without autocorrelation), and there is no seasonality: the seasonal strength (0.004, 0.065, 0.115 for 5, 12 and 30 days) is what a simulated random walk produces.
- None of the 7 coefficients of the ARIMA(5, 1, 2) is significant (p-values above 0.1), and the AIC chooses the orders (0, 1, 0).
- No forecast is checked against real prices.
- The reference imports `statsmodels.tsa.arima_model.ARIMA`, which no longer exists in current statsmodels.

## Our approach

1. Tested stationarity (augmented Dickey-Fuller) and measured the seasonality against a simulated random walk.
2. Fitted the reference ARIMA and looked at the significance of its coefficients.
3. Evaluated five forecasts of 20 trading days with an expanding window for the last value, the drift, the ARIMA of the reference, an ARIMA chosen with the AIC and the seasonal SARIMA of the reference (tools: `seasonal_decompose`, `ARIMA`, `SARIMAX`).

## Results

| Model | MAE ($) | RMSE ($) | MAPE (%) |
|---|---|---|---|
| ARIMA(5, 1, 2), reference | 6.40 | 7.22 | 5.19 |
| Drift | 6.43 | 7.26 | 5.21 |
| Last value (random walk) | 6.47 | 7.30 | 5.23 |
| ARIMA(0, 1, 0), chosen with the AIC | 6.47 | 7.30 | 5.23 |
| SARIMA(5, 1, 2)x(5, 1, 2, 12), reference | 8.34 | 9.32 | 6.65 |

The ARIMA of the reference is not better than repeating the last price, and the seasonal SARIMA is 28% worse in RMSE. See the notebook's Conclusion for details.
