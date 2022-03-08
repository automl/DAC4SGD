import dataclasses
import json
from typing import List, Union

import numpy as np
import torch.optim

from dac4automlcomp.policy import DACPolicy, DeterministicPolicy


class Serializable:
    def save(self, path):
        file_path = path / f"{self.__class__.__name__}.json"
        with file_path.open(mode="w") as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, path):
        file_path = path / f"{cls.__name__}.json"
        with file_path.open(mode="r") as f:
            return cls(**json.load(f))


@dataclasses.dataclass
class ConstantLRPolicy(Serializable, DeterministicPolicy, DACPolicy):
    lr: float

    def act(self, _):
        return self.lr

    def reset(self, instance):
        pass


@dataclasses.dataclass
class CosineAnnealingLRPolicy(Serializable, DeterministicPolicy, DACPolicy):
    lr: float

    def act(self, state):
        return 0.5 * (1 + np.cos(state["step"] * np.pi / self.cutoff)) * self.lr

    def reset(self, instance):
        self.cutoff = instance.cutoff


@dataclasses.dataclass
class SimplePolicy(Serializable, DeterministicPolicy, DACPolicy):
    lr: float
    a: float
    b: float

    def act(self, state):
        self.loss += state["loss"].sum()
        if not (state["step"] + 1) % self.epoch_size:
            if self.prev_loss is not None:
                self.lr_t *= self.a if self.loss > self.prev_loss else 1 / self.b
            self.prev_loss = self.loss
            self.loss = 0.0
        return self.lr_t

    def reset(self, instance):
        self.lr_t = self.lr
        self.loss = 0.0
        self.prev_loss = None
        self.epoch_size = len(instance.loaders[0])


@dataclasses.dataclass
class ReduceLROnPlateauPolicy(Serializable, DeterministicPolicy, DACPolicy):
    lr: float
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: Union[float, List[float]] = 0
    eps: float = 1e-8

    def act(self, state):
        if state["validation_loss"] is not None:
            self.scheduler.step(state["validation_loss"].mean())
        return self.optimizer.param_groups[0]["lr"]

    def reset(self, _):
        self.scheduler, self.optimizer = self.__create_scheduler(
            **dataclasses.asdict(self)
        )

    @staticmethod
    def __create_scheduler(*, lr, **scheduler_params):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )
        return scheduler, optimizer
