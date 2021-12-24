from collections import namedtuple
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import numpy as np


Instance = namedtuple(
    "Instance", ["model", "optimizer_params", "loss", "loaders", "steps"]
)


def random_feature_extractor(rng: np.random.RandomState, **kwargs) -> nn.Module:
    conv1 = rng.randint(low=16, high=129)
    conv2 = rng.randint(low=16, high=129)
    return nn.Sequential(
        nn.Conv2d(1, conv1, 3, 1),
        nn.ReLU(),
        nn.Conv2d(conv1, conv2, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
    )


def random_mnist_model(rng: np.random.RandomState, **kwargs) -> nn.Module:
    class MNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = random_feature_extractor(rng)
            shape = self.f(torch.zeros((1, 1, 28, 28))).shape
            n_features = torch.prod(torch.tensor(shape))
            self.fc1 = nn.Linear(n_features, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.50)

        def forward(self, x):
            x = self.f(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    return MNISTModel()


def random_mnist_loader(
    rng: np.random.RandomState, **kwargs
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_kwargs = {"batch_size": kwargs["training_batch_size"]}
    test_kwargs = {"batch_size": kwargs["validation_batch_size"]}
    dataset1 = datasets.MNIST("data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("data", train=False, transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader


def random_optimizer_parameters(rng, **kwargs):
    lr = kwargs["learning_rate"].sample(rng)
    return {"lr": lr}


def random_mnist_instance(rng: np.random.RandomState, **kwargs) -> Instance:
    model = random_mnist_model(rng, **kwargs)
    loaders = random_mnist_loader(rng, **kwargs)
    optimizer_params = random_optimizer_parameters(rng, **kwargs)
    loss = F.nll_loss
    steps = kwargs["steps"].sample(rng)
    return Instance(model, optimizer_params, loss, loaders, steps)


def random_instance(rng: np.random.RandomState, **kwargs) -> Instance:
    datasets = ["MNIST"]
    idx = rng.randint(low=0, high=len(datasets))
    dataset = datasets[idx]
    if dataset == "MNIST":
        instance = random_mnist_instance(rng, **kwargs)
    else:
        raise NotImplementedError
    return instance
