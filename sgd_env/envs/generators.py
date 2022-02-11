import sys
from collections import namedtuple
from functools import lru_cache, partial
from typing import Any, Tuple

if sys.version_info.minor >= 8:
    from typing import Protocol
else:
    from typing_extensions import Protocol  # type: ignore

import numpy as np
import torch
import torch.nn.functional as F
from ConfigSpace import (
    AndConjunction,
    ConfigurationSpace,
    Constant,
    GreaterThanCondition,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


SGDInstance = Instance("SGDInstance", [
        "dataset",
        "model",
        "optimizer_params",
        "loss",
        "batch_size",
        "loaders",
        "cutoff",
        "crash_penalty"],
)


@lru_cache(maxsize=None)
def default_configuration_space() -> ConfigurationSpace:
    cs = ConfigurationSpace()
    cutoff = UniformIntegerHyperparameter("cutoff", 300, 900)

    modify = UniformFloatHyperparameter("modify", 0, 1)

    optimizer_parameters = [
        UniformFloatHyperparameter("eps", 1e-10, 1e-6, log=True),
        UniformFloatHyperparameter("weight_decay", 0.0001, 1, log=True),
        UniformFloatHyperparameter("beta1", 0.0001, 0.2, log=True),
        UniformFloatHyperparameter("beta2", 0.0001, 0.2, log=True),
    ]

    batch_size_exp = UniformIntegerHyperparameter("batch_size_exp", 2, 8, log=True)
    train_val = Constant("train_validation_ratio", 0.9)

    cs.add_hyperparameters(
        [cutoff, batch_size_exp, train_val, modify, *optimizer_parameters]
    )

    for param in optimizer_parameters:
        mod = UniformFloatHyperparameter(f"mod_{param.name}", 0, 1)
        cs.add_hyperparameter(mod)
        cs.add_condition(
            AndConjunction(
                GreaterThanCondition(param, modify, 0.8),
                GreaterThanCondition(param, mod, 0.5),
            )
        )
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
    n_features = int(
        torch.prod(torch.tensor(f(torch.zeros((1, 1, 28, 28))).shape)).item()
    )
    return nn.Sequential(f, nn.Flatten(1), nn.Linear(n_features, 10), nn.LogSoftmax(1))


def random_mlp_mnist_model(rng: np.random.RandomState, **kwargs) -> nn.Module:
    l1 = 2 ** (8 - int(np.exp(rng.uniform(low=np.log(1), high=np.log(3)))))
    l2 = 2 ** (7 - int(np.exp(rng.uniform(low=np.log(1), high=np.log(4)))))
    return nn.Sequential(
        nn.Flatten(1),
        nn.Linear(784, l1),
        nn.ReLU(),
        nn.Linear(l1, l2),
        nn.ReLU(),
        nn.Linear(l2, 10),
        nn.LogSoftmax(1),
    )


def random_mnist_loader(rng: np.random.RandomState, **kwargs) -> Tuple[DataLoader, Any]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_kwargs = {"batch_size": 2 ** kwargs["batch_size_exp"]}
    val_kwargs = {"batch_size": 64}
    dataset1 = datasets.MNIST("data", train=True, download=True, transform=transform)
    if "dataset_subset" in kwargs:
        dataset1 = torch.utils.data.Subset(dataset1, kwargs["dataset_subset"])
    train_size = int(len(dataset1) * kwargs["train_validation_ratio"])
    train, val = torch.utils.data.random_split(
        dataset1, [train_size, len(dataset1) - train_size]
    )
    train_loader = DataLoader(train, **train_kwargs)
    val_loader = DataLoader(val, **val_kwargs)
    return (train_loader, val_loader)


def random_optimizer_parameters(rng, **kwargs):
    return {
        "betas": (1 - kwargs.get("beta1", 0.1), 1 - kwargs.get("beta2", 0.0001)),
        "weight_decay": kwargs.get("weight_decay", 1e-2),
        "eps": kwargs.get("eps", 1e-8),
    }


def random_mnist_instance(rng: np.random.RandomState, **kwargs):
    model_types = ["CNN", "MLP"]
    model_type = model_types[rng.randint(low=0, high=len(model_types))]
    if model_type == "CNN":
        model = random_mnist_model(rng, **kwargs)
    elif model_type == "MLP":
        model = random_mlp_mnist_model(rng, **kwargs)
    else:
        raise NotImplementedError
    batch_size = 2 ** kwargs["batch_size_exp"]
    loaders = random_mnist_loader(rng, **kwargs)
    optimizer_params = random_optimizer_parameters(rng, **kwargs)
    loss = F.nll_loss
    cutoff = kwargs["cutoff"]
    crash_penalty = np.log(0.1) * cutoff
    return ["MNIST", model, optimizer_params, loss, batch_size, loaders, cutoff, crash_penalty]


def random_instance(
    rng: np.random.RandomState, cs: ConfigurationSpace, **kwargs
) -> Instance:
    datasets = ["MNIST"]
    default_rng_state = torch.get_rng_state()
    seed = rng.randint(1, 4294967295, dtype=np.int64)
    cs.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = cs.sample_configuration()
    idx = rng.randint(low=0, high=len(datasets))
    dataset = datasets[idx]
    if dataset == "MNIST":
        instance = random_mnist_instance(rng, **config, **kwargs)
    else:
        raise NotImplementedError
    torch.set_rng_state(default_rng_state)
    SGDInstance.i = (dataset, *instance)
    return SGDInstance


default_instance_generator: GeneratorFunc = partial(
    random_instance, cs=default_configuration_space()
)
