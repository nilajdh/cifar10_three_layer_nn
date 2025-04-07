import matplotlib.pyplot as plt

from model.load_cifar10 import CIFAR10Dataloader
from model.mlp_model import MLPModel
from model.loss import CrossEntropyLoss
from model.optimizer import SGDOptimizer
from model.trainer import Trainer

# 神经网络超参
nn_arch = [
    {"input_dim": 3072, "output_dim": 784, "activation_type": "relu"},
    {"input_dim": 784, "output_dim": 32, "activation_type": "relu"},
    {"input_dim": 32, "output_dim": 10, "activation_type": "softmax"},
]

dataloader_kwargs = {
    "path_dir": "data/cifar-10-batches-py",
    "n_valid": 2000,
    "batch_size": 32,
}

optimizer_kwargs = {
    "lr": 0.05,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 6000,
}

trainer_kwargs = {
    "n_epochs": 100,
    "eval_step": 5
}


def main():
    dataloader = CIFAR10Dataloader(**dataloader_kwargs)
    model = MLPModel(nn_arch, lambda_l2=0.0001)
    optimizer = SGDOptimizer(**optimizer_kwargs)
    loss = CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss, dataloader, **trainer_kwargs)
    trainer.train(save_ckpt=True, verbose=True)
    trainer.save_log("logs/")
    trainer.save_best_model("checkpoints/", metric="loss", n=3, keep_last=True)
    trainer.clear_cache()

    plt.show(block=True)


if __name__ == "__main__":
    main()
