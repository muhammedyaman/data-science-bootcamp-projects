# Image Classification with TensorFlow

Deep learning project: recognizing 10 types of clothing in 28 x 28 grayscale images (Fashion-MNIST).

## Dataset

Fashion-MNIST (Zalando Research, MIT licence): 60,000 training and 10,000 test images, loaded in the notebook with `tf.keras.datasets.fashion_mnist.load_data()`, so no data file is included.

## Reference solution

[Image Classification with TensorFlow in Machine Learning](https://amanxai.com/2020/08/30/image-classification-with-tensorflow-in-machine-learning/) (Aman Kharwal): one hidden Dense layer of 128 neurons, 10 epochs, test accuracy 0.882.

Issues found: no validation set, one training only (no idea of the variation between trainings), no result per class, no comparison with a convolutional network.

## Our approach

1. Reproduced the reference network and trained it three times with different seeds.
2. Trained a convolutional network the same way, with a validation split and early stopping.
3. Compared them on the test set and looked at the classification report and confusion matrix.

## Results

| Model | Test accuracy (mean of 3 seeds) | Std |
|---|---|---|
| Dense network (reference) | 0.875 | 0.003 |
| Convolutional network | 0.906 | 0.002 |

The shirt class is the weakest one (F1 0.70), mostly confused with T-shirt, coat and pullover. See the notebook's Conclusion.
