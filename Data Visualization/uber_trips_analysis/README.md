# Uber Trips Analysis

Data visualization project: when and where Uber pickups happen in New York City (September 2014).

## Dataset

`uber-raw-data-sep14.csv` from the FiveThirtyEight repository [uber-tlc-foil-response](https://github.com/fivethirtyeight/uber-tlc-foil-response) (`uber-trip-data/uber-raw-data-sep14.csv`): 1,028,136 pickups with time, latitude, longitude and base code. The file is 47 MB and is not stored in this repository; download it into this folder to run the notebook. The copy that the reference links to is no longer available.

## Reference solution

[Uber Trips Analysis using Python](https://amanxai.com/2021/04/21/uber-trips-analysis-using-python/) (Aman Kharwal): density plots (`distplot`) of the day, weekday and hour, a map of the pickups, and five conclusions (Monday is the most profitable day, fewer people use Uber on Saturdays, 6 pm is the busiest hour, trips rise from 5 am, most pickups are near Manhattan).

Issues found:
- The reference reads the weekday 0 as Sunday, but `dt.weekday` starts with 0 = Monday (1 September 2014 was a Monday), so every weekday conclusion is shifted by one day.
- Totals by weekday are inflated for Monday and Tuesday, which occur five times in September 2014 (the other days four times); counts must be compared per day.
- Density curves are drawn through whole numbers (days, hours).

## Our approach

1. Parsed the time, added the weekday name (0 = Monday) and checked the data (24,037 exact duplicate rows kept; 1.4% of the pickups outside the city area, left out of the map).
2. Plotted the pickups per day of the month, per weekday (total and per day) and per weekday and hour (heatmap), the density of pickups on a map (hexagonal bins) and the share of each base.

## Results

| Day | Pickups per day |
|---|---|
| Saturday | 40,514 |
| Friday | 40,095 |
| Thursday | 38,319 |
| Wednesday | 33,843 |
| Tuesday | 32,646 |
| Sunday | 29,133 |
| Monday | 27,458 (includes the Labor Day holiday) |

Saturday and Friday are the busiest days, not Monday; 6 pm is the busiest hour (75,040 pickups); most pickups are in Manhattan, with a hot spot at JFK airport. See the notebook's Conclusion for details.
