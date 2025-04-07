import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-12
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        :param y_pred: 输出标签(samples, classes)
        :param y_true: 真实标签 (samples, classes)
        :return: 交叉熵损失
        """
        self.y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        self.y_true = y_true
        return -np.sum(self.y_true * np.log(self.y_pred), axis=1).mean()

    def backward(self):
        grad = (self.y_pred - self.y_true) / self.y_pred.shape[0]
        return grad
