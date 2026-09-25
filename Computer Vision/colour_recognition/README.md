# Colour Recognition

Computer vision project: naming a colour from its RGB values, and naming the main colours of a photo.

## Data

- `colors.csv`: 865 colour names with RGB values ([source](https://github.com/amankharwal/Website-data/blob/master/colors.csv)); 100 rows repeat the RGB value of another row.
- `colour.jpeg`: a painting with seven colour stripes ([source](https://github.com/amankharwal/Website-data/blob/master/colour.jpeg)).

## Reference solution

[Colour Recognition with Python](https://amanxai.com/2020/09/14/colour-recognition-with-python/) (Thecleverprogrammer / Aman Kharwal): an OpenCV window where a double click on a pixel returns the colour name, found with a Python loop over the table using the sum of absolute RGB differences.

Issues found:
- The tool is interactive (mouse callback), so it cannot be run or tested in a notebook.
- The `for` loop with `table.loc[i, ...]` is very slow (4.75 s for 100 pixels, against 0.004 s for a nearest-neighbour search).
- Ties are resolved in favour of the last row (`d <= minimum`), and 100 colours share an RGB value with another name, so the answer depends on the row order.
- The accuracy of the returned names is never measured.

## Our approach

1. Kept one name per RGB value (765 colours) and wrote the search as a 1-nearest-neighbour classifier.
2. Compared three distance measures (Manhattan in RGB as in the reference, Euclidean in RGB, Euclidean in Lab) by adding noise to each table colour, once in RGB and once in Lab, and checking whether the original colour is found again.
3. Found the dominant colours of the photo with K-Means and named the cluster centres.

## Results

| Noise model | Noise level | Manhattan RGB | Euclidean RGB | Euclidean Lab |
|---|---|---|---|---|
| RGB | 5 | 0.801 | 0.805 | 0.692 |
| RGB | 10 | 0.485 | 0.495 | 0.403 |
| Lab | 1 | 0.873 | 0.878 | 0.932 |
| Lab | 2 | 0.650 | 0.646 | 0.800 |

Each measure wins when the noise matches its own space, so no distance can be called better without human-labelled colour names. The photo's seven stripes are found and named (golden yellow, teal, orange, Dodger Blue, candy apple red, dark royal blue, violet-red). See the notebook's Conclusion for details.
