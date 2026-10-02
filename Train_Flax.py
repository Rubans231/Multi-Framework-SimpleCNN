import struct
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

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

RAW = "./data/FashionMNIST/raw"


def load_images(path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    return data.astype(np.float32) / 255.0


def load_labels(path):
    with open(path, "rb") as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)


x_train = load_images(f"{RAW}/train-images-idx3-ubyte")
y_train = load_labels(f"{RAW}/train-labels-idx1-ubyte")
x_test = load_images(f"{RAW}/t10k-images-idx3-ubyte")
y_test = load_labels(f"{RAW}/t10k-labels-idx1-ubyte")
