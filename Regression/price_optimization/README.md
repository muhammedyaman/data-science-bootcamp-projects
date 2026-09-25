# Price Optimization

Regression project: modeling the relationship between price and demand (item quantity), and using it to find a revenue-maximizing price.

## Dataset

`Competition_Data.csv` - 100,000 rows, each representing a store/item/week/competitor-price observation. Columns: `Fiscal_Week_ID`, `Store_ID`, `Item_ID`, `Price`, `Item_Quantity`, `Sales_Amount_No_Discount`, `Sales_Amount`, `Competition_Price`.

The raw data has repeated rows per (Store_ID, Item_ID, Fiscal_Week_ID) combination (one row per competitor price observation). This was aggregated to one row per combination (8,800 rows) before any analysis, to avoid pseudoreplication.

## Reference solution

[thecleverprogrammer.com/amanxai.com Price Optimization](https://thecleverprogrammer.com/) - computes elasticity with `pct_change()` on raw, non-grouped data, buckets items into Low/Medium/High price tiers by average price, and applies a fixed rule (+5% for Medium tier, -10% for High tier) to every item in a tier. No model, no train/test split, no evaluation metric.

Issues found in the reference approach (see notebook for details):
- `pct_change()` applied without grouping/sorting mostly compares repeated rows (constant price) or unrelated items across group boundaries, producing meaningless elasticity values.
- The fixed percentage rule is applied uniformly to every item in a price tier, regardless of whether that specific item's demand actually responds to price.

## Our approach

1. Aggregated the data to remove pseudoreplication.
2. Reproduced the reference's `pct_change()` elasticity calculation (both the naive buggy version and a corrected, grouped/sorted version) to show why a ratio-based elasticity is unstable.
3. Fit a constant-elasticity demand model (`ln(Q) = ln(a) + b*ln(P)`) instead of the ratio-based approach.
4. Checked whether a pooled elasticity estimate is trustworthy: item-level correlation tests (176 items) showed only ~5% were statistically significant, so a pooled `LinearRegression` with `Item_ID` and `Store_ID` fixed effects (dummy variables) was fit to isolate the true within-item price effect. Result: price has no statistically meaningful effect on demand once item/store baselines are controlled for (coefficient ~0.008).
5. Restricted price optimization to the 5 items with a statistically significant (p<0.05) negative price-quantity correlation, fit a per-item log-log model for each, and scanned price only within the observed range (no extrapolation) to find the revenue-maximizing price.

## Results

Pooled fixed-effects regression (test set): R2 = 0.963, RMSE = 0.029, MAE = 0.025 (on log scale); `log_Price` coefficient = 0.008 (not meaningfully different from zero).

Per-item price scan (5 statistically significant items), revenue-maximizing price found within the observed range:

| Item | Elasticity (b) | Current price | Optimal price | Revenue lift |
|---|---|---|---|---|
| item_739 | -0.46 | 163.78 | 173.10 | +3.06% |
| item_911 | -0.41 | 119.38 | 123.82 | +2.18% |
| item_410 | -0.45 | 222.00 | 239.88 | +4.31% |
| item_877 | -0.33 | 283.70 | 310.66 | +6.27% |
| item_338 | -0.33 | 113.13 | 115.52 | +1.41% |

All 5 items are demand-inelastic (|b| < 1), so the revenue-maximizing price found is a boundary solution at the top of the observed price range, not a true interior optimum. See the notebook's Conclusion section for a full discussion, including where and why this diverges from the reference solution.
