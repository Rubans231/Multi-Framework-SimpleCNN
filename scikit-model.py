import numpy as np
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(
    root="./data", train=True, download=False, transform=transform
)
test_data = datasets.FashionMNIST(
    root="./data", train=False, download=False, transform=transform
)


def flatten_dataset(dataset):
    images = np.array([np.array(img).flatten() for img, _ in dataset])
    labels = np.array([label for _, label in dataset])
    return images, labels


print("Flattening images (28x28 to 784-len vectors)")
x_train, y_train = flatten_dataset(train_data)
x_test, y_test = flatten_dataset(test_data)
print(f"x_train shape: {x_train.shape}")


#  Train

print("\nTraining LogisticRegression...")
clf = LogisticRegression(max_iter=200)
clf.fit(x_train, y_train)


# Evaluate

y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nLogistic Regression test accuracy: {accuracy:.2f}%")

print("\n--- Per class report ---")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
