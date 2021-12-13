from dataclasses import field
from typing import List, Tuple, Type

import torch

from .dataclass_config import Config
from .generators import InstanceGeneratorFunc, random_instance_generator


default_config = Config()


@default_config('dac')
class DACConfig:
    control: List[str] = field(default_factory=lambda: ['lr'])
    n_instances: int = 100


@default_config('optimizer')
class OptimizerConfig:
    optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW

    # User settings
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.01
    amsgrad: bool = False


@default_config('generator')
class GeneratorConfig:
    generator_func: InstanceGeneratorFunc = random_instance_generator

    # User settings
    epoch_range: Tuple[int, int] = (300, 900)
    learning_rate_range: Tuple[float, float] = (0.0001, 1.0)
    training_batch_size: int = 64
    validation_batch_size: int = 64


