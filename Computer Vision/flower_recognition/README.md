# Flower Recognition

Computer vision project: recognizing the type of a flower (daisy, dandelion, rose, sunflower, tulip) from a photo.

## Dataset

The TensorFlow [flower_photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) set: 3,670 Flickr photos in five classes (dandelion 898, tulips 799, sunflowers 699, roses 641, daisy 633), about 320 x 240 pixels. It is about 230 MB and is not stored in this repository: download the archive and extract it to a `flower_photos/` folder next to the notebook. The reference uses the Kaggle [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) set (4,242 images, chamomile instead of daisy), which requires a login, so the numbers cannot be compared directly.

## Reference solution

[Flower Recognition with Python](https://amanxai.com/2020/11/24/flower-recognition-with-python/) (Thecleverprogrammer / Aman Kharwal): a CNN (64, 128, 128, 128 filters and two dense layers) trained for 64 epochs with data augmentation, on 128 x 128 images.

Issues found:
- The test set is used as `validation_data` during training, so the test set takes part in the training decisions.
- The result is judged by looking at a grid of 36 predicted images; no accuracy, F1 or confusion matrix.
- `model.predict(X_test)` is called twice per image of the grid (72 full predictions).
- No transfer learning, although only about 4,000 images are available.

## Our approach

1. Read and resized the images; stratified 80/20 split; the test set is used only for the final scores, 10% of the training images serve as validation.
2. Trained a small CNN, the same network with augmentation, and a VGG16 model with a frozen pre-trained base and a new head.
3. Trained the reference network for 10 epochs to compare (64 epochs take more than two hours on this CPU).
4. Compared accuracy and macro F1, and looked at the confusion matrix and the wrongly classified images.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Small CNN | 0.646 | 0.638 |
| Small CNN with augmentation | 0.783 | 0.781 |
| VGG16 transfer learning | 0.828 | 0.826 |
| Reference network, 10 of 64 epochs | 0.616 | 0.596 |

The small CNN overfits (0.999 training accuracy); augmentation and especially transfer learning help. See the notebook's Conclusion for details and limitations.
