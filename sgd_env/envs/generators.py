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
        "cutoff",
        "crash_penalty",
    ],
)


class GeneratorFunc(Protocol):
    def __call__(self, rng: np.random.RandomState, **kwargs: int) -> Instance:
        ...


@lru_cache(maxsize=None)
def default_configuration_space() -> ConfigurationSpace:
    cs = ConfigurationSpace()
    cutoff = UniformIntegerHyperparameter("cutoff", 300, 900)
    learning_rate = UniformFloatHyperparameter("lr", 0.0001, 0.1, log=True)
    batch_size_exp = UniformIntegerHyperparameter("batch_size_exp", 2, 16, log=True)
    cs.add_hyperparameters([cutoff, learning_rate, batch_size_exp])
    return cs


def random_feature_extractor(rng: np.random.RandomState, **kwargs) -> nn.Module:
    conv1 = int(np.exp(rng.uniform(low=np.log(2), high=np.log(9))))
    conv2 = int(np.exp(rng.uniform(low=np.log(6), high=np.log(24))))
    conv3 = int(np.exp(rng.uniform(low=np.log(32), high=np.log(256))))
    return nn.Sequential(
        nn.Conv2d(1, conv1, 3, 1),
        nn.MaxPool2d(2),
        nn.Conv2d(conv1, conv2, 3, 1),
        nn.MaxPool2d(2),
        nn.Conv2d(conv2, conv3, 3, 1),
        nn.ReLU(),
    )


def random_mnist_model(rng: np.random.RandomState, **kwargs) -> nn.Module:
    f = random_feature_extractor(rng)
    n_features = torch.prod(torch.tensor(f(torch.zeros((1, 1, 28, 28))).shape))
    return nn.Sequential(f, nn.Flatten(1), nn.Linear(n_features, 10), nn.LogSoftmax(1))


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
    batch_size = 2 ** kwargs["batch_size_exp"]
    loaders = random_mnist_loader(rng, batch_size=batch_size)
    optimizer_params = random_optimizer_parameters(rng, **kwargs)
    loss = F.nll_loss
    cutoff = kwargs["cutoff"]
    crash_penalty = np.log(0.1) * cutoff
    return model, optimizer_params, loss, batch_size, loaders, cutoff, crash_penalty


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
