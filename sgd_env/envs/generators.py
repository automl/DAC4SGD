from collections import namedtuple
from typing import Tuple, Any, Protocol
from functools import partial, lru_cache

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter
from ConfigSpace import UniformFloatHyperparameter


Instance = namedtuple(
    "Instance",
    [
        "dataset",
        "model",
        "optimizer_params",
        "loss",
        "batch_size",
        "loaders",
        "steps",
        "crash_penalty",
    ],
)


class GeneratorFunc(Protocol):
    def __call__(self, rng: np.random.RandomState, **kwargs: int) -> Instance:
        ...


@lru_cache(maxsize=None)
def default_configuration_space() -> ConfigurationSpace:
    cs = ConfigurationSpace()
    steps = UniformIntegerHyperparameter("steps", 300, 900)
    learning_rate = UniformFloatHyperparameter("lr", 0.0001, 0.1, log=True)
    batch_size = UniformIntegerHyperparameter("batch_size", 32, 256, log=True)
    cs.add_hyperparameters([steps, learning_rate, batch_size])
    return cs


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


def random_mnist_loader(rng: np.random.RandomState, **kwargs) -> Tuple[DataLoader, Any]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_kwargs = {"batch_size": kwargs["batch_size"]}
    dataset1 = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    return (train_loader, None)


def random_optimizer_parameters(rng, **kwargs):
    return {"lr": kwargs["lr"]}


def random_mnist_instance(rng: np.random.RandomState, **kwargs):
    model = random_mnist_model(rng, **kwargs)
    batch_size = kwargs["batch_size"]
    loaders = random_mnist_loader(rng, batch_size=batch_size)
    optimizer_params = random_optimizer_parameters(rng, **kwargs)
    loss = F.nll_loss
    steps = kwargs["steps"]
    crash_penalty = np.log(0.1) * steps
    return model, optimizer_params, loss, batch_size, loaders, steps, crash_penalty


def random_instance(rng: np.random.RandomState, cs: ConfigurationSpace) -> Instance:
    datasets = ["MNIST"]
    cs.seed(rng.randint(1, 4294967295, dtype=np.int64))
    config = cs.sample_configuration()
    idx = rng.randint(low=0, high=len(datasets))
    dataset = datasets[idx]
    if dataset == "MNIST":
        instance = random_mnist_instance(rng, **config)
    else:
        raise NotImplementedError
    return Instance(dataset, *instance)


default_instance_generator: GeneratorFunc = partial(
    random_instance, cs=default_configuration_space()
)
