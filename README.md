# Multi-Framework SimpleCNN

Training on FashionMNIST accessed through torchvision datasets to make a SimpleCNN capable of classification and identification of the categories of items it was familiarized on.

Ported the same concept over to various different Frameworks and benchmarked each to find variance between their performance.

Framework        Test accuracy
PyTorch          87.85–89.44% (varies by run)
TensorFlow/Keras 89.79%
Flax/JAX (NNX)   89.34%

## Results

### Pytorch CNN

![CNN-result](assets/screenshot_20260923_104602.png)

### Scikit

![Scikit-result](assets/screenshot_20260923_104940.png)

### Tensorflow Keras

![](assets/screenshot_20260928_180843.png)

### Transfer Learning

![](assets/screenshot_20261002_075148.png)
