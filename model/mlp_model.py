import json
import pickle

import numpy as np
import model.layer as L


class MLPModel:
    def __init__(self, nn_arch=None, lambda_l2=0.01):
        self.layers = []
        self.nn_arch = nn_arch if nn_arch else []
        self.lambda_l2 = lambda_l2
        for layer in nn_arch:
            self.layers.append(L.Linear(layer["input_dim"], layer["output_dim"]))
            self.layers.append(L.Activation(layer["activation_type"]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for i, layer in enumerate(reversed(self.layers)):
            if isinstance(layer, L.Linear):
                dw, db, grad = layer.backward(grad, layer.input_cache, self.lambda_l2)
                layer.dw = dw
                layer.db = db
            elif isinstance(layer, L.Activation):
                grad = layer.backward(grad, layer.input_cache)
        return grad

    def inference(self, x):
        return np.argmax(self.forward(x), axis=1)

    def deep_copy(self):
        """
        :return: model 拷贝
        """
        model_copy = MLPModel(self.nn_arch)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, L.Linear):
                model_copy.layers[i].w = layer.w.copy()
                model_copy.layers[i].b = layer.b.copy()
        return model_copy

    def save_model_dict(self, path):
        """
        保存模型结构和参数
        :param path: .pkl 文件的保存路径
        """
        model_dict = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, L.Linear):
                model_dict[f"layer_{i}_w"] = layer.w
                model_dict[f"layer_{i}_b"] = layer.b
        with open(path, "wb") as f:
            pickle.dump(model_dict, f)
        with open(path.replace(".pkl", ".json"), "w") as f:
            json.dump(self.nn_arch, f)

    def load_model_dict(self, path):
        """
        加载模型结构和参数
        :param path: .pkl 文件的加载路径
        """
        with open(path.replace(".pkl", ".json"), "r") as f:
            nn_arch = json.load(f)
        self.__init__(nn_arch)
        with open(path, "rb") as f:
            model_dict = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, L.Linear):
                layer.w = model_dict[f"layer_{i}_w"]
                layer.b = model_dict[f"layer_{i}_b"]
                layer.zero_grad()
