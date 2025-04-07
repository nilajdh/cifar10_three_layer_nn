import numpy as np


class Layer:
    def __init__(self):
        self.input_cache = None


class Activation(Layer):
    """ 激活函数可以从 relu, sigmoid, softmax, leakyrelu 中选择 """
    def __init__(self, activation_type):
        super().__init__()
        self.type = activation_type
        self.leaky_alpha = 0.01 if activation_type == "leakyrelu" else None

    def forward(self, x):
        self.input_cache = x
        if self.type == "relu":
            return np.maximum(0, x)
        if self.type == "sigmoid":
            return 1 / (1 + np.exp(-x))
        if self.type == "softmax":
            # 避免数值溢出令x减去其中最大值
            expx = np.exp(x - np.max(x, axis=1, keepdims=True))
            return expx / np.sum(expx, axis=1, keepdims=True)
        if self.type == "leakyrelu":
            return np.where(x > 0, x, x * self.leaky_alpha)

    def backward(self, grad, x):
        # grad 表示损失函数关于当前激活层的梯度
        if self.type == "relu":
            return grad * (x > 0)
        if self.type == "leakyrelu":
            dx = np.zeros_like(x)
            dx[x < 0] = self.leaky_alpha
            return grad * dx
        if self.type == "sigmoid" or self.type == "softmax":
            sig = self.forward(x)
            return grad * sig * (1 - sig)


class Linear(Layer):
    """ 构建线性层 """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w = np.random.normal(0, pow(input_dim, -0.5), (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.input_cache = x
        return np.dot(x, self.w) + self.b

    def backward(self, grad, x, lambda_l2):
        dw = np.dot(x.T, grad) + lambda_l2 * self.w
        db = np.sum(grad, axis=0, keepdims=True)
        dx = np.dot(grad, self.w.T)
        return dw, db, dx

    def zero_grad(self):
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
