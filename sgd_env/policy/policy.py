import dataclasses
from abc import ABC, abstractmethod
from typing import Optional
import json

import numpy as np

from sgd_env import generators


class AbstractPolicy(ABC):
    @abstractmethod
    def act(self, state):
        ...

    @abstractmethod
    def reset(self, instance: generators.Instance):
        ...

    @abstractmethod
    def save(self, f):
        ...

    @classmethod
    @abstractmethod
    def load(cls, f):
        ...


class Serializable:
    def save(self, f):
        json.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, f):
        return cls(**json.load(f))


@dataclasses.dataclass
class ConstantLRPolicy(Serializable, AbstractPolicy):
    lr: float

    def act(self, _):
        return self.lr

    def reset(self, _):
        pass


@dataclasses.dataclass
class CosineAnnealingLRPolicy(Serializable, AbstractPolicy):
    lr: float

    def act(self, state):
        return 0.5 * (1 + np.cos(state["step"] * np.pi / self.steps)) * self.lr

    def reset(self, instance: generators.Instance):
        self.steps = instance.steps


@dataclasses.dataclass
class SimplePolicy(Serializable, AbstractPolicy):
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

    def reset(self, instance: generators.Instance):
        self.lr_t = self.lr
        self.loss = 0.0
        self.prev_loss = None
        self.epoch_size = len(instance.loaders[0])
