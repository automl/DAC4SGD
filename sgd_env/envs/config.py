from dataclasses import field
from typing import List, Tuple, Type, Callable, Optional

import torch
import numpy as np
from gym.spaces import Space, Dict, Box

from .dataclass_config import Config
from .generators import InstanceGeneratorFunc, random_instance
from .hyperparameters import Hyperparameter
from .hyperparameters import UniformFloatHyperparameter
from .hyperparameters import UniformIntegerHyperparameter


default_config = Config()

Actions = List[Tuple[str, Space, Callable[[torch.optim.Optimizer, str, Dict], None]]]


def optimizer_action(optimizer, name, action):
    for g in optimizer.param_groups:
        g[name] = action[name]


@default_config("dac")
class DACConfig:
    actions: Actions = field(
        default_factory=lambda: [
            ("lr", Box(low=-np.inf, high=np.inf, shape=(1,)), optimizer_action)
        ]
    )
    n_instances: int = np.inf
    device: str = "cpu"
    seed: Optional[int] = None


@default_config("optimizer")
class OptimizerConfig:
    optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW

    # User settings
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.01
    amsgrad: bool = False


@default_config("generator")
class GeneratorConfig:
    generator_func: InstanceGeneratorFunc = random_instance

    # User settings
    steps: Hyperparameter = UniformIntegerHyperparameter(300, 900)
    learning_rate: Hyperparameter = UniformFloatHyperparameter(0.0001, 0.1, True)
    training_batch_size: int = 64
    validation_batch_size: int = 64
