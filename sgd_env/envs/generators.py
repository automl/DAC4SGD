from collections import namedtuple
from dataclasses import InitVar, dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ConfigSpace import (
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.hyperparameters import Hyperparameter
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from dac4automlcomp.dac_env import Generator

SGDInstance = namedtuple(
    "SGDInstance",
    [
        "dataset",
        "model",
        "optimizer_params",
        "loss",
        "batch_size",
        "train_validation_ratio",
        "loaders",
        "cutoff",
        "crash_penalty",
    ],
)


@dataclass
class DefaultSGDGenerator(Generator[SGDInstance]):
    cutoff: InitVar[Hyperparameter] = UniformIntegerHyperparameter("cutoff", 300, 900)
    batch_size_exp: InitVar[Hyperparameter] = UniformIntegerHyperparameter(
        "batch_size_exp", 2, 8, log=True
    )
    validation_train_percent: InitVar[Hyperparameter] = UniformIntegerHyperparameter(
        "validation_train_percent", 1, 20, log=True, default_value=10
    )
    eps: InitVar[Hyperparameter] = UniformFloatHyperparameter(
        "eps",
        1e-10,
        1e-6,
        log=True,
        default_value=1e-8,
    )
    weight_decay: InitVar[Hyperparameter] = UniformFloatHyperparameter(
        "weight_decay",
        0.0001,
        1,
        log=True,
        default_value=1e-2,
    )
    inv_beta1: InitVar[Hyperparameter] = UniformFloatHyperparameter(
        "inv_beta1",
        0.0001,
        0.2,
        log=True,
        default_value=0.1,
    )
    inv_beta2: InitVar[Hyperparameter] = UniformFloatHyperparameter(
        "inv_beta2",
        0.0001,
        0.2,
        log=True,
        default_value=0.0001,
    )

    def __post_init__(self, *args):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameters(args)

    def random_instance(self, rng):
        default_rng_state = torch.get_rng_state()
        seed = rng.randint(1, 4294967295, dtype=np.int64)
        self.cs.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        config = self.cs.sample_configuration()
        dataset_types = ["MNIST"]
        idx = rng.randint(low=0, high=len(dataset_types))
        dataset = dataset_types[idx]
        if dataset == "MNIST":
            instance = self._random_mnist_instance(rng, **config)
        else:
            raise NotImplementedError
        torch.set_rng_state(default_rng_state)
        return SGDInstance(dataset, *instance)

    def _sample_optimizer_params(self, rng, **kwargs):
        modify = rng.rand()
        samples = {
            parameter: (
                kwargs[parameter]
                if modify > 0.8 and rng.rand() > 0.5
                else getattr(self, parameter).default_value
            )
            for parameter in [
                "weight_decay",
                "eps",
                "inv_beta1",
                "inv_beta2",
            ]
        }

        return {
            "weight_decay": samples["weight_decay"],
            "eps": samples["eps"],
            "betas": (1 - samples["inv_beta1"], 1 - samples["inv_beta2"]),
        }

    @staticmethod
    def _random_feature_extractor(rng: np.random.RandomState, **kwargs) -> nn.Module:
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

    def _random_mnist_model(self, rng: np.random.RandomState, **kwargs) -> nn.Module:
        f = self._random_feature_extractor(rng)
        n_features = int(
            torch.prod(torch.tensor(f(torch.zeros((1, 1, 28, 28))).shape)).item()
        )
        return nn.Sequential(
            f, nn.Flatten(1), nn.Linear(n_features, 10), nn.LogSoftmax(1)
        )

    @staticmethod
    def _random_mlp_mnist_model(rng: np.random.RandomState, **kwargs) -> nn.Module:
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

    @staticmethod
    def _random_mnist_loader(
        rng: np.random.RandomState, **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset1 = datasets.MNIST(
            "data", train=True, download=True, transform=transform
        )
        test = datasets.MNIST("data", train=False, transform=transform)
        train_validation_ratio = 1 - kwargs["validation_train_percent"] / 100
        train_size = int(len(dataset1) * train_validation_ratio)
        train, val = torch.utils.data.random_split(
            dataset1, [train_size, len(dataset1) - train_size]
        )
        train_loader = DataLoader(train, batch_size=2 ** kwargs["batch_size_exp"])
        val_loader = DataLoader(val, batch_size=64)
        test_loader = DataLoader(test, batch_size=64)
        return (train_loader, val_loader, test_loader)

    def _random_mnist_instance(self, rng: np.random.RandomState, **kwargs):
        model_types = ["CNN", "MLP"]
        model_type = model_types[rng.randint(low=0, high=len(model_types))]
        if model_type == "CNN":
            model = self._random_mnist_model(rng, **kwargs)
        elif model_type == "MLP":
            model = self._random_mlp_mnist_model(rng, **kwargs)
        else:
            raise NotImplementedError
        batch_size = 2 ** kwargs["batch_size_exp"]
        loaders = self._random_mnist_loader(rng, **kwargs)
        optimizer_params = self._sample_optimizer_params(rng, **kwargs)
        loss = F.nll_loss
        cutoff = kwargs["cutoff"]
        crash_penalty = np.log(0.1) * cutoff
        train_validation_ratio = 1 - kwargs["validation_train_percent"] / 100
        return (
            model,
            optimizer_params,
            loss,
            batch_size,
            train_validation_ratio,
            loaders,
            cutoff,
            crash_penalty,
        )
