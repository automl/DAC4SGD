from collections import namedtuple
from dataclasses import InitVar, dataclass
from typing import Tuple
import os

import numpy as np
import torch
import torch.nn.functional as F
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.hyperparameters import Hyperparameter
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from dac4automlcomp.generator import Generator
from sgd_env.envs.utils import surpress_output

datasets.CIFAR10.download = surpress_output(datasets.CIFAR10.download)


DATASETS = {
    "MNIST": {
        "transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    },
    "CIFAR10": {
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    },
    "FashionMNIST": {
        "transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    },
}

"""
Target Problem Instance definition: A supervised neural network training task.
All properties of the task can be used inside the DAC policy (and is passed on reset) with exception of the test_loader.
"""
SGDInstance = namedtuple(
    "SGDInstance",
    [
        "dataset",  # Name of the dataset
        "model",  # Initialized torch.nn.Module model
        "optimizer_params",  # kwargs for optimizer initializing excluding parameters
        "loss",  # Callable loss function with torch.nn.functional API
        "batch_size",  # Step-wise gradient estimates are based on batch_size data points
        "train_validation_ratio",  # Train dataset size / validation dataset size
        "fraction_of_dataset",  # Used fraction of full dataset
        "loaders",  # train loader, validation_loader, test_loader
        "cutoff",  # Number of optimization steps
        "crash_penalty",  # Received reward when target algorithm crashes (typical by divergence)
    ],
)


@dataclass
class DefaultSGDGenerator(Generator[SGDInstance]):
    batch_size_exp: InitVar[Hyperparameter] = UniformIntegerHyperparameter(
        "batch_size_exp", 4, 8, log=True
    )
    validation_train_percent: InitVar[Hyperparameter] = UniformIntegerHyperparameter(
        "validation_train_percent", 5, 20, log=True, default_value=10
    )
    fraction_of_dataset: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "fraction_of_dataset", [1, 0.5, 0.2, 0.1]
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
    dataset_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    def __post_init__(self, *args):
        """Initialize configuration space using `InitVar` arguments of the class."""
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameters(args)

    def random_instance(self, rng: np.random.RandomState):
        """Samples a random `SGDInstance`"""
        default_rng_state = torch.get_rng_state()
        seed = rng.randint(1, 4294967295, dtype=np.int64)
        self.cs.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        config = self.cs.sample_configuration()
        dataset = rng.choice(np.array(list(DATASETS.keys())))
        instance = self._random_instance(rng, dataset, **config)
        torch.set_rng_state(default_rng_state)
        return instance

    def _sample_optimizer_params(self, rng, **kwargs):
        """Samples optimizer parameters according to below rules.
        - With 0.8 probability keep default of all parameters
        - For each hyperparameter, with 0.5 probability sample a new value else keep default
        """
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

    def _random_architecture(
        self,
        rng: np.random.RandomState,
        input_shape: Tuple[int, int, int],
        n_classes: int,
        **kwargs
    ) -> nn.Module:
        """Samples random architecture with `rng` for given `input_shape` and `n_classes`."""
        modules = [nn.Identity()]
        max_n_conv_layers = 3
        n_conv_layers = rng.randint(low=0, high=max_n_conv_layers + 1)
        prev_conv = input_shape[0]
        kernel_sizes = [3, 5, 7][: max(0, 3 - n_conv_layers + 1)]

        for layer_idx, layer_exp in enumerate(range(1, int(n_conv_layers * 2 + 1), 2)):
            conv = int(
                np.exp(
                    rng.uniform(
                        low=np.log(2 ** layer_exp), high=np.log(2 ** (layer_exp + 2))
                    )
                )
            )
            kernel_size = rng.choice(kernel_sizes)
            modules.append(nn.Conv2d(prev_conv, conv, kernel_size, 1))
            modules.append(nn.MaxPool2d(2))
            prev_conv = conv

        activation = rng.choice([nn.Identity, nn.ReLU, nn.PReLU, nn.ELU])
        batch_norm = rng.choice([nn.Identity, nn.BatchNorm2d])
        if n_conv_layers:
            modules.pop()
            modules.append(activation())
            modules.append(batch_norm(prev_conv))
        feature_extractor = nn.Sequential(*modules)
        linear_layers = [nn.Flatten()]
        max_n_mlp_layers = 2
        n_mlp_layers = int(rng.randint(low=0, high=max_n_mlp_layers + 1))
        prev_l = int(
            torch.prod(
                torch.tensor(feature_extractor(torch.zeros((1, *input_shape))).shape)
            ).item()
        )
        for layer_idx in range(n_mlp_layers):
            l = 2 ** (
                2 ** (max_n_mlp_layers + 1 - layer_idx)
                - int(
                    np.exp(
                        rng.uniform(
                            low=np.log(1), high=np.log(max_n_mlp_layers + 1 + layer_idx)
                        )
                    )
                )
            )
            linear_layers.append(nn.Linear(prev_l, l))
            linear_layers.append(nn.ReLU())
            prev_l = l

        linear_layers.append(nn.Linear(prev_l, n_classes))
        linear_layers.append(nn.LogSoftmax(1))
        mlp = nn.Sequential(*linear_layers)
        return nn.Sequential(feature_extractor, mlp)

    def _random_torchvision_loader(
        self, rng: np.random.RandomState, name: str, **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, test loaders for `name` dataset."""
        transform = DATASETS[name]["transform"]
        train_dataset = getattr(datasets, name)(
            self.dataset_path, train=True, download=True, transform=transform
        )
        train_size = int(len(train_dataset) * kwargs["fraction_of_dataset"])
        classes = train_dataset.classes
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [train_size, len(train_dataset) - train_size]
        )
        train_dataset.classes = classes
        test = getattr(datasets, name)(
            self.dataset_path, train=False, transform=transform
        )
        train_validation_ratio = 1 - kwargs["validation_train_percent"] / 100
        train_size = int(len(train_dataset) * train_validation_ratio)
        train, val = torch.utils.data.random_split(
            train_dataset, [train_size, len(train_dataset) - train_size]
        )
        train_loader = DataLoader(train, batch_size=2 ** kwargs["batch_size_exp"])
        val_loader = DataLoader(val, batch_size=64)
        test_loader = DataLoader(test, batch_size=64)
        return (train_dataset, test), (train_loader, val_loader, test_loader)

    def _random_instance(self, rng: np.random.RandomState, dataset: str, **kwargs):
        """Samples an SGDInstance instance for given `dataset`."""
        batch_size = 2 ** kwargs["batch_size_exp"]
        datasets, loaders = self._random_torchvision_loader(rng, dataset, **kwargs)
        model = self._random_architecture(
            rng, datasets[0][0][0].shape, len(datasets[0].classes)
        )
        optimizer_params = self._sample_optimizer_params(rng, **kwargs)
        loss = F.nll_loss
        n_params = len(torch.nn.utils.parameters_to_vector(model.parameters()))
        target_runtime = 60
        epoch_cutoff = max(
            1,
            int(
                batch_size
                / len(datasets[0])
                * (target_runtime - 0.5)
                / (0.01 + 0.00035 * batch_size + (0.0000002 * n_params) ** 2)
            ),
        )
        cutoff = int(len(loaders[0]) * epoch_cutoff)

        crash_penalty = np.log(len(datasets[0].classes))
        train_validation_ratio = 1 - kwargs["validation_train_percent"] / 100
        return SGDInstance(
            dataset,
            model,
            optimizer_params,
            loss,
            batch_size,
            train_validation_ratio,
            kwargs["fraction_of_dataset"],
            loaders,
            cutoff,
            crash_penalty,
        )
