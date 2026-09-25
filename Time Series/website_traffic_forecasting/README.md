# Website Traffic Forecasting

Time series project: forecasting the daily views of a website.

## Dataset

[Thecleverprogrammer.csv](https://raw.githubusercontent.com/amankharwal/Website-data/master/Thecleverprogrammer.csv): 391 daily values of the views of thecleverprogrammer.com, from 1 June 2021 to 26 June 2022 (mean 8,483; dates written day first). Included in this folder.

## Reference solution

[Website Traffic Forecasting using Python](https://amanxai.com/2022/06/28/website-traffic-forecasting-using-python/) (Aman Kharwal): a decomposition with a period of 30 days, a SARIMA model with the orders (5, 1, 2) and the seasonal orders (5, 1, 2, 12) fitted on the whole series, and a forecast of the next 50 days.

Issues found:
- The traffic has a weekly cycle (the article says that weekdays are busier), but neither the decomposition (30 days) nor the SARIMA model (12 days) uses a weekly period; the seasonal strength is 0.65 for 7 days, 0.07 for 30 and 0.01 for 12.
- The 15 or more parameters of the model are chosen by eye from plots.
- No forecast is compared with real values.

## Our approach

1. Parsed the dates day first and explored the weekly pattern and the autocorrelation.
2. Measured the strength of the seasonality for the periods 7, 12 and 30 with the seasonal decomposition.
3. Evaluated five forecasts of 28 days with an expanding window for the seasonal naive forecast, the weekday mean, the reference SARIMA, a weekly SARIMA whose orders were chosen with the AIC, and Prophet.

## Results

| Method | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| SARIMA (1, 1, 1) x (1, 1, 1, 7), weekly period | 714.5 | 914.9 | 7.7 |
| Mean of each weekday over the last 4 weeks | 792.6 | 1,013.8 | 8.4 |
| Prophet, default settings | 883.5 | 1,081.5 | 9.8 |
| Seasonal naive | 910.3 | 1,130.5 | 9.7 |
| SARIMA of the reference | 932.0 | 1,179.4 | 10.2 |

The weekly SARIMA is 22% better than the model of the reference in RMSE, and the reference model is worse than repeating the last week. See the notebook's Conclusion for details.
