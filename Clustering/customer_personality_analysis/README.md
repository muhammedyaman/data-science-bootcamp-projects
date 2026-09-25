# Customer Personality Analysis

Clustering project: segmenting customers by income, spending and seniority.

## Dataset

`marketing_campaign.csv` (2,240 customers, 29 columns: demographics, product spending, purchase channels and campaign responses), from [this link](https://raw.githubusercontent.com/amankharwal/Website-data/master/marketing_campaign.csv) used by the reference. Included in this folder.

## Reference solution

[Customer Personality Analysis with Python](https://amanxai.com/2021/02/08/customer-personality-analysis-with-python/) (Aman Kharwal / Thecleverprogrammer): a Gaussian Mixture with 4 components on standardized, row-normalized income, seniority and spending; the components are then named "Stars", "Need attention", "High potential" and "Leaky bucket"; the article continues with the Apriori algorithm.

Issues found:
- The component numbers of a mixture model are arbitrary, but the reference assigns the segment names to them without checking the profiles. In our run of its code, none of the four names matches its component.
- Customers with a birth year of 1893 to 1900 (age above 110) are not removed; the number of components is fixed without any check.
- The Apriori part (association rules) is not part of the course and is not reproduced.

## Our approach

1. Cleaned missing and extreme incomes, impossible ages and constant columns (2,212 customers left).
2. Rebuilt the same three variables and reproduced the reference's mixture model to compare names and profiles.
3. Standardized the variables, chose k with the elbow method and silhouette score, and clustered with K-Means; compared with Agglomerative clustering.
4. Named the segments from their profile and validated them with the campaign response, which was not used for the clustering.

## Results

| Segment | Customers | Income | Seniority (months) | Spending | Campaign response |
|---|---|---|---|---|---|
| Stars | 480 | 71,078 | 21 | 1,318 | 29.4% |
| High potential | 439 | 73,801 | 9 | 1,102 | 13.4% |
| Leaky bucket | 641 | 36,382 | 20 | 220 | 15.1% |
| Need attention | 652 | 38,491 | 9 | 132 | 5.5% |

Silhouette: K-Means 0.372, reference Gaussian Mixture 0.370, Agglomerative 0.279. See the notebook's Conclusion for details and limitations.
