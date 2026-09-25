# Financial Budget Analysis

Data visualization project: how the Union Budget of India 2021 is divided between the ministries.

## Dataset

[India_budget_2021.csv](https://raw.githubusercontent.com/amankharwal/Website-data/master/India_budget_2021.csv): the fund allotted (in crores of rupees) to 55 ministries and departments, an empty row and a grand total of 3,483,235.63 crores. Included in this folder.

## Reference solution

[Financial Budget Analysis with Python](https://amanxai.com/2021/04/05/financial-budget-analysis-with-python/) (Aman Kharwal): nine ministries selected by row number, a typed value for "OTHERS", a bar chart and a donut chart; conclusion: the Ministry of Finance gets 40% of the funds.

Issues found:
- The typed value of "OTHERS" (592,971.08) is 15,699.65 crores too low, exactly the allotment of the Ministry of Micro, Small and Medium Enterprises, which is therefore missing from the chart; percentages are slightly too large (Finance 40.0% instead of 39.8%).
- Ministries are picked by row numbers and the rest is typed by hand instead of being computed.
- `dropna()` is not assigned, so the empty row and the grand total stay in the table.

## Our approach

1. Separated the grand total from the ministries and checked that the ministries add up to it.
2. Reproduced the reference selection and located the missing amount.
3. Ranked the ministries and drew a bar chart of the ten largest with their shares, a donut chart with the nine largest and the computed rest, the cumulative share curve, and the small allotments on a logarithmic scale.

## Results

| Ministry | Fund (crores) | Share |
|---|---|---|
| Finance | 1,386,273 | 39.8% |
| Defence | 478,196 | 13.7% |
| Consumer Affairs | 256,948 | 7.4% |
| Home Affairs | 166,547 | 4.8% |
| Rural Development | 133,690 | 3.8% |
| Agriculture | 131,531 | 3.8% |
| Other 49 ministries | 930,051 | 26.7% |

Two ministries hold 53.5% of the budget, nine hold 80% and thirteen hold 90%; 41 of the 55 ministries have less than 1% each. See the notebook's Conclusion for details.
