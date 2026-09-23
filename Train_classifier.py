import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(
    root="./data", train=True, download=False, transform=transform
)
test_data = datasets.FashionMNIST(
    root="./data", train=False, download=False, transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

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


# The Model


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Train

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}\n avg training loss: {avg_loss:.4f}")

# test

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        all_preds.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\nOverall accuracy: {accuracy:.2f}%")

print("\n--- Per-class resport---")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

print("--- Confusion matrix ---")
cm = confusion_matrix(all_labels, all_preds)
header = "       " + " ".join(f"{i:4d}" for i in range(10))
print(header)
for i, row in enumerate(cm):
    print(f"true {i}: " + " ".join(f"{v:4d}" for v in row))
print(
    "\n(class indices: "
    + ", ".join(f"{i}={name}" for i, name in enumerate(CLASS_NAMES))
    + ")"
)
