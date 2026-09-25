# Billionaires Analysis

Data visualization project: who the billionaires of the world are, where they live, in which industries they made their fortune and how old they are (Forbes ranking of 2021).

## Dataset

[Billionaire.csv](https://raw.githubusercontent.com/amankharwal/Website-data/master/Billionaire.csv): 2,755 billionaires with name, net worth (text such as `$177 B`), country, source of wealth, rank, age (79 missing) and industry. Included in this folder.

## Reference solution

[Billionaires Analysis with Python](https://amanxai.com/2021/06/24/billionaires-analysis-with-python/) (Aman Kharwal): removes every row with a missing age, converts the net worth, draws a chart of the ten richest people and donut charts of the five countries, sources and industries with the most billionaires; conclusion: the United States and China have the most billionaires, which is read as a sign of their business environment.

Issues found:
- The chart of the ten richest people (`histplot` of the names coloured by net worth) shows ten bars of the same height and no information.
- Removing the 79 rows without an age is not needed for the counts and changes them (Germany 136 to 115, China 626 to 610, India 140 to 134), so the order of the top five countries changes; the missing ages are not random (75% of the billionaires of the United Arab Emirates).
- The number of billionaires is read as a measure of the business environment without any population or economy size to compare with.

## Our approach

1. Converted the net worth, kept all the rows (the age is only ignored in the age charts) and checked the repeated names (different people).
2. Reproduced the reference chart and the effect of the removal on the counts.
3. Drew the ten richest people, the wealth concentration, the countries (count, total and median net worth), the industries (count and net worth on a logarithmic scale) and the age (distribution, by industry, against net worth).

## Results

| Country | Billionaires | Total net worth (billion dollars) | Median net worth |
|---|---|---|---|
| United States | 724 | 4,398 | 2.8 |
| China | 626 | 2,532 | 2.0 |
| India | 140 | 596 | 2.2 |
| Germany | 136 | 626 | 2.8 |
| Russia | 118 | 586 | 2.4 |
| Hong Kong | 71 | 448 | 3.6 |

The United States and China hold 49% of the billionaires and 53% of the wealth. The top 10 hold 8.8% of the total wealth (median fortune 2.3 against a mean of 4.7 billion dollars); the median age is 63, and age has almost no relation to net worth. See the notebook's Conclusion for details.
