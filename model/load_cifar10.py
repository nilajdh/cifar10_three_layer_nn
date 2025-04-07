import pickle
import numpy as np
import os


class CIFAR10Dataloader:
    def __init__(self, path_dir="data/cifar-10-batches-py", n_valid=1000, batch_size=32):
        X_train, y_train = self.load_cifar10_train(path_dir)
        self.x_train, self.y_train, self.x_valid, self.y_valid = self.train_valid_split(X_train, y_train, n_valid)
        self.x_test, self.y_test = self.load_cifar10_test(path_dir)
        self.batch_size = batch_size

    @staticmethod
    def load_cifar10_train(path_dir):
        X = []
        y = []
        for i in range(1, 6):
            file_path = os.path.join(path_dir, f'data_batch_{i}')
            with open(file_path, 'rb') as f:
                datadict = pickle.load(f, encoding='bytes')
                X_batch = datadict[b'data']
                y_batch = datadict[b'labels']
                X.extend(X_batch)
                y.extend(y_batch)
        X = np.array(X)
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        X = X.reshape(X.shape[0], -1)
        X = X / 255.0
        y = np.eye(10)[y]
        return X, y

    @staticmethod
    def load_cifar10_test(path_dir):
        file_path = os.path.join(path_dir, 'test_batch')
        with open(file_path, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data']
            y = datadict[b'labels']
        X = np.array(X)
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        X = X.reshape(X.shape[0], -1)
        X = X / 255.0
        y = np.eye(10)[y]
        return X, y

    @staticmethod
    def train_valid_split(x_train, y_train, n_valid):
        n_samples = x_train.shape[0]
        indices = np.random.permutation(n_samples)
        valid_indices = indices[:n_valid]
        train_indices = indices[n_valid:]
        return x_train[train_indices], y_train[train_indices], x_train[valid_indices], y_train[valid_indices]

    def generate_train_batch(self):
        n_samples = self.x_train.shape[0]
        indices = np.random.permutation(self.x_train.shape[0])
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_train[batch_indices], self.y_train[batch_indices]

    def generate_valid_batch(self):
        n_samples = self.x_valid.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_valid[batch_indices], self.y_valid[batch_indices]

    def generate_test_batch(self):
        n_samples = self.x_test.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_test[batch_indices], self.y_test[batch_indices]
