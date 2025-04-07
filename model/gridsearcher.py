import itertools
from tqdm import tqdm

from model.load_cifar10 import CIFAR10Dataloader
from model.mlp_model import MLPModel
from model.loss import CrossEntropyLoss
from model.optimizer import SGDOptimizer
from model.trainer import Trainer


class GridSearcher:
    def __init__(self, opts, defaults, lambda_l2):
        self.combinations = self.generate_combinations(opts, defaults)
        self.lambda_l2 = lambda_l2
        self.results = []

    @staticmethod
    def generate_combinations(hyper_param_opts, hyper_param_defaults):
        """
        生成超参数组合
        :param hyper_param_opts: 可选的超参数表
        :param hyper_param_defaults: 默认的超参数表
        """
        for key in hyper_param_opts.keys():
            if len(hyper_param_opts[key]) == 0:
                hyper_param_opts.pop(key)
        for key in hyper_param_defaults.keys():
            if key not in hyper_param_opts.keys() or len(hyper_param_opts[key]) == 0:
                hyper_param_opts[key] = [hyper_param_defaults[key]]

        combinations = []
        for values in itertools.product(*hyper_param_opts.values()):
            combination = dict(zip(hyper_param_opts.keys(), values))
            combinations.append(combination)
        return combinations

    @staticmethod
    def generate_config(combination):
        """
        根据超参数组合生成网络结构
        :param combination: 超参数组合
        """
        n_layers = sum([1 for key in combination.keys() if "hidden_size" in key]) + 1
        nn_arch = []
        if n_layers == 1:
            layer = {
                "input_dim": 3072,
                "output_dim": 10,
                "activation_type": combination["activation_1"],
            }
            nn_arch.append(layer)
        elif n_layers > 1:
            layer = {
                "input_dim": 3072,
                "output_dim": combination["hidden_size_1"],
                "activation_type": combination["activation_1"],
            }
            nn_arch.append(layer)
            for i in range(1, n_layers - 1):
                layer = {
                    "input_dim": combination[f"hidden_size_{i}"],
                    "output_dim": combination[f"hidden_size_{i + 1}"],
                    "activation_type": combination[f"activation_{i + 1}"],
                }
                nn_arch.append(layer)
            layer = {
                "input_dim": combination[f"hidden_size_{n_layers - 1}"],
                "output_dim": 10,
                "activation_type": combination[f"activation_{n_layers}"],
            }
            nn_arch.append(layer)

        lr_list = combination["lr"] if isinstance(combination["lr"], list) else [combination["lr"]]
        ld_list = combination["ld"] if isinstance(combination["ld"], list) else [combination["ld"]]
        optimizer_kwargs_list = []
        for lr in lr_list:
            for ld in ld_list:
                optimizer_kwargs = {
                    "lr": lr,
                    "ld": ld,
                    "decay_rate": combination["decay_rate"],
                    "decay_step": combination["decay_step"],
                }
                optimizer_kwargs_list.append(optimizer_kwargs)
        return nn_arch, optimizer_kwargs_list

    def search(self, dataloader_kwargs, trainer_kwargs, metric="loss"):
        for combination in tqdm(self.combinations):
            nn_arch, optimizer_kwargs_list = self.generate_config(combination)
            dataloader = CIFAR10Dataloader(**dataloader_kwargs)
            model = MLPModel(nn_arch, self.lambda_l2)
            for optimizer_kwargs in optimizer_kwargs_list:
                optimizer = SGDOptimizer(**optimizer_kwargs)
                loss = CrossEntropyLoss()

                trainer = Trainer(model, optimizer, loss, dataloader, **trainer_kwargs)
                trainer.train(save_ckpt=False, verbose=False)
                valid_loss, valid_acc = trainer.evaluate()
                self.results.append((combination, valid_loss, valid_acc))

        if metric == "loss":
            self.results.sort(key=lambda x: x[1])
        elif metric == "acc":
            self.results.sort(key=lambda x: x[2], reverse=True)
        return self.results
