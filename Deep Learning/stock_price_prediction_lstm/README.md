# Stock Price Prediction with LSTM

Deep learning project: forecasting the next day's closing price of the Apple stock with LSTM networks.

## Dataset

`apple_stock.csv`: 3,449 trading days of Apple (AAPL), from 23 April 2008 to 31 December 2021 (open, high, low, close, adjusted close, volume), downloaded with `yfinance` as in the reference. The prices are adjusted for the share splits of 2014 and 2020.

## Reference solution

[Stock Price Prediction with LSTM](https://amanxai.com/2022/01/03/stock-price-prediction-with-lstm/) (Aman Kharwal): the close of a day is "predicted" from the open, high, low and volume of the same day; random 80/20 split; a two-layer LSTM with batch size 1 and 30 epochs; only the training loss is shown.

Issues found:
- The high and the low of a day contain its close: a linear regression on the same features reaches an R2 of 0.9999 and a 0.49% error, so nothing is forecast.
- The random split puts the test days between the training days.
- The LSTM receives four features as four time steps, the features are not scaled (volume up to 2.6 billion), and the batch size of 1 makes the loss jump between 2 and 9.
- No error is computed on test data.

## Our approach

1. Showed with a linear regression that the reference task is trivial.
2. Built a real forecast: the previous 60 days predict the next day; split in time order (training until November 2017, validation until December 2019, a test period of 518 days); scaling fitted on the training days; early stopping.
3. Compared LSTMs on scaled prices and on returns with naive baselines (yesterday's close, drift, the mean of the last 5 days).

## Results

| Forecast of the next day's close | MAE ($) | RMSE ($) | MAPE (%) | Direction correct (%) |
|---|---|---|---|---|
| Last value (yesterday's close) | 1.75 | 2.39 | 1.61 | not defined |
| Drift | 1.75 | 2.39 | 1.61 | 54.3 |
| Mean of the last 5 days | 2.50 | 3.32 | 2.25 | 51.7 |
| LSTM on scaled prices | 4.09 | 4.95 | 3.32 | 45.6 |
| LSTM on returns | 1.75 | 2.39 | 1.60 | 55.2 |

The LSTM on prices is 2.3 times worse than the naive forecast (the test prices are far above the training range); the LSTM on returns equals the naive forecast and finds no signal in the past prices. See the notebook's Conclusion for details.
