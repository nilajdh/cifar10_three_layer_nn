import json

from model.gridsearcher import GridSearcher

hyper_param_defaults = {
    "input_dim": 3072,
    "hidden_size_1": 784,
    "hidden_size_2": 32,
    "output_dim": 10,
    "activation_1": "relu",
    "activation_2": "relu",
    "activation_3": "softmax",
    "lr": 0.05,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 6000,
}

dataloader_kwargs = {
    "path_dir": "data/cifar-10-batches-py",
    "n_valid": 2000,
    "batch_size": 32,
}

trainer_kwargs = {
    "n_epochs": 20,
    "eval_step": 30,  # 默认在搜索超参组合中不评估
}


def main():
    hyper_param_opts = {
        "hidden_size_1": [256, 768, 1536],
        "hidden_size_2": [128, 64, 32],
        "lr": [0.05, 0.01],
        "ld": [0.001, 0.005],
    }
    searcher = GridSearcher(hyper_param_opts, hyper_param_defaults, lambda_l2=0.0001)
    results = searcher.search(dataloader_kwargs, trainer_kwargs, metric="loss")
    with open("gridsearch_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
