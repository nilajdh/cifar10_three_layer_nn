## Introduction
Manually build a three-layer neural network classifier and train it on the CIFAR-10 dataset to achieve image classification.

## Structure
Please place the other files in the following structure:
```
code_root/
    ├── checkpoints/
    │    ├── with_l2
    │    └── without_l2
    ├── data/
    │    └── cifar-10-batches-py
    └── logs
```

## Train and Evaluate
Run the following codes to train the network. You can also change the hyperparameters in the file:
```
python train_mlp_cifar10.py
```
Run the following codes to test the network. You can also change the structure(.json) and checkpoint(.pkl) in the file:
```
python test_mlp_cifar10.py
```

## Parameters Search
Run the following codes to search for the optimal hyperparameters. You can also change the optical parameters in the file:
```
python gridsearch_mlp_cifar10.py
```
