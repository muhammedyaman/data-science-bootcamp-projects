# Count Objects in Image

Computer vision project: counting the objects of one class (cars, people, chairs, ...) in a photo with a pre-trained object detector.

## Data

Seven photos in `images/`: `cars.jpg`, `dogs.jpg`, `g8.jpg`, `kitchen.jpeg` and `bunchofshapes.jpg` come from the course material; `bus.jpg` and `zidane.jpg` are the sample images delivered with the `ultralytics` package. The pre-trained detector `yolov8n.pt` is included; the small and medium weights (`yolov8s.pt`, `yolov8m.pt`) are downloaded automatically by `ultralytics` on the first run. The objects of each photo were counted by eye to build a ground truth (see the notebook).

## Reference solution

[Count Objects in Image using Python](https://amanxai.com/2021/05/11/count-objects-in-image-using-python/) (Aman Kharwal): the `cvlib` library (YOLOv3) on one motorway photo, the class `car` is counted; the article prints 10 cars.

Issues found:
- The number is not compared with the real number of cars (11 by eye in the same photo).
- One photo, one confidence threshold, one class; no error measure.
- It depends on an old detector and library; readers reported problems running it (missing model files).

## Our approach

1. Used the YOLOv8 detector (`ultralytics`) and counted the boxes of the wanted class with `value_counts`, as in the lesson.
2. Compared the counts with the counts by eye on seven photos (nine image and class pairs) for three model sizes (nano, small, medium) and four confidence thresholds.
3. Compared counting only `car` with counting all vehicles, and tried a classical OpenCV method (threshold and contours) on a simple drawing and on the motorway photo.

## Results

Mean absolute error of the count over nine targets:

| Model | Threshold 0.10 | 0.25 | 0.50 | 0.70 |
|---|---|---|---|---|
| Nano | 2.00 | 0.44 | 0.56 | 1.11 |
| Small | 1.11 | 0.22 | 0.22 | 1.00 |
| Medium | 0.67 | 0.22 | 0.22 | 1.00 |

On the motorway photo the class `car` gives 11 (the reference reports 10). The threshold matters: too low adds false objects, too high misses real ones. Contours count simple shapes on a white background (4 of 4) but fail on a real photo (16 regions for 11 cars). See the notebook's Conclusion for details.
