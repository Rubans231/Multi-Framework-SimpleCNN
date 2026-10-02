import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
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
base_train_idx = [
    i for i, label in enumerate(y_train.tolist()) if label in BASE_CLASSES
]
base_train = Subset(full_train, base_train_idx)
base_train_loader = DataLoader(base_train, batch_size=64, shuffle=True)
print(f"Pretraining set: {len(base_train)} images across {len(BASE_CLASSES)} classes")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


criterion = nn.CrossEntropyLoss()

print(
    "\n ---------------------- Pretraining on 8 out of 10 classes ----------------------"
)
base_model = SimpleCNN(num_classes=8)
optimizer = optim.Adam(base_model.parameters(), lr=0.001)

for epoch in range(3):
    base_model.train()
    running_loss = 0.0
    for images, labels in base_train_loader:
        optimizer.zero_grad()
        loss = criterion(base_model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"epoch: {epoch + 1}/3")
    print(f"loss: {running_loss / len(base_train_loader):.4f}")

# ---------------------- Transfer Learning on the pretrained layers ----------------------


for param in base_model.conv1.parameters():
    param.requires_grad = False
for param in base_model.conv2.parameters():
    param.requires_grad = False

base_model.fc1 = nn.Linear(32 * 7 * 7, 128)
base_model.fc2 = nn.Linear(128, 10)

full_train_loader = DataLoader(full_train, batch_size=64, shuffle=True)
full_test_loader = DataLoader(full_test, batch_size=64, shuffle=False)

transfer_optimizer = optim.Adam(
    list(base_model.fc1.parameters()) + list(base_model.fc2.parameters()),
    lr=0.001,
)

base_model.train()
running_loss = 0.0
for images, labels in full_train_loader:
    transfer_optimizer.zero_grad()
    loss = criterion(base_model(images), labels)
    loss.backward()
    transfer_optimizer.step()
    running_loss += loss.item()
print(
    "---------------------- Post-training with Transfer Learning ----------------------"
)
print(f"loss: {running_loss / len(full_train_loader):.4}")

base_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in full_test_loader:
        preds = base_model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
transfer_acc = 100 * correct / total
print(f"Test accuracy: {transfer_acc:.2f}%")
