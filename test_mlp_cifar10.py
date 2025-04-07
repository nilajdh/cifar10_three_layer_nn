import numpy as np
import json

from model.load_cifar10 import CIFAR10Dataloader
from model.loss import CrossEntropyLoss
from model.mlp_model import MLPModel

dataloaders_kwargs = {
    "path_dir": "data/cifar-10-batches-py",
    "batch_size": 32,
}

f = open('checkpoints/with_l2/model_epoch_100.json', 'r')
content = f.read()
nn_arch = json.loads(content)
ckpt_path = "checkpoints/with_l2/model_epoch_100.pkl"


def main():
    dataloader = CIFAR10Dataloader(**dataloaders_kwargs)
    model = MLPModel(nn_arch, lambda_l2=0)
    model.load_model_dict(ckpt_path)
    loss = CrossEntropyLoss()

    total_loss = 0
    total_acc = 0

    for x_batch, y_batch in dataloader.generate_test_batch():
        y_pred = model.forward(x_batch)
        total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
        ce_loss = loss.forward(y_pred, y_batch)
        total_loss += ce_loss * len(x_batch)

    test_loss = total_loss / len(dataloader.y_test)
    test_acc = total_acc / len(dataloader.y_test)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Checkpoint: {ckpt_path} | ")


if __name__ == "__main__":
    main()
