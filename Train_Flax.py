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


class SimpleCNN(nnx.Module):
    def __init__(self, num_classes: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 16, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.fc1 = nnx.Linear(32 * 7 * 7, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, num_classes, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nnx.relu(self.conv2(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.fc1(x))


model = SimpleCNN(num_classes=10, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


def loss_fn(model, x, y):
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss, logits
