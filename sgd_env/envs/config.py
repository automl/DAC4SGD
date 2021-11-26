from dataclasses import field
from typing import List, Tuple, Type

import torch

from .dataclass_config import Config
from .generators import InstanceGeneratorFunc, random_instance_generator


default_config = Config()

@default_config('generator')
class GeneratorConfig:
    generator_func: InstanceGeneratorFunc = random_instance_generator

    # User settings
    datasets: List[str] = field(default_factory=lambda: ['MNIST'])
    epoch_range: Tuple[int, int] = (300, 900)
    learning_rate_range: Tuple[float, float] = (0.0001, 1.0)
    optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW
    training_batch_size: int = 64
    validation_batch_size: int = 64


