import struct
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

Raw = "./data/FashionMNIST/raw"


def load_images(path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def load_labels(path):
    with open(path, "rb") as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)


x_train = load_images(f"{Raw}/train-images-idx3-ubyte")
y_train = load_images(f"{Raw}/train-labels-idx1-ubyte")
x_test = load_images(f"{Raw}/t10k-images-idx3-ubyte")
y_test = load_images(f"{Raw}/t10k-labels-idx1-ubyte")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
print("x_train shape:", x_train.shape, "| x_test shape:", x_test.shape)

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10),
    ]
)
model.summary()
