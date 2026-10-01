import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset

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

RAW = "./data/FashionMNIST/raw/"

def load_images(path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    return torch.tensor(data, dtype=torch.float32).unsqueeze(1) / 255.0

def load_labels(path):
    with open(path, "rb") as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.tensor(data, dtype=torch.long)

x_train = load_images(f"{RAW}/train-images-idx3-ubyte")
y_train = load_labels(f"{RAW}/train-labels-idx1-ubyte")
x_test = load_images(f"{RAW}/t10k-images-idx3-ubyte")
y_test = load_labels(f"{RAW}/t10k-labels-idx1-ubyte")

full_train = TensorDataset(x_train, y_train)
full_test = TensorDataset(x_test, y_test)

BASE_CLASSES = list(range(8))
base_train_idx = [i for i, label in enumerate(y_train.tolist()) if label in BASE_CLASSES]
base_train = Subset(full_train, base_train_idx)
base_train_loader = DataLoader(base_train, batch_size=64, shuffle=True)
print(f"Phase A training set: {len(base_train)} images across {len(BASE_CLASSES) classes}")
