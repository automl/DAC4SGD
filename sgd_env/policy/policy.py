import dataclasses
from abc import ABC, abstractmethod
from ast import literal_eval
from typing import Optional

import numpy as np

from sgd_env import generators


class AbstractPolicy(ABC):
    @abstractmethod
    def act(self, observation):
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


@dataclasses.dataclass
class ConstantLRPolicy(AbstractPolicy):
    lr: float

    def __post_init__(self):
        assert self.lr > 0

    def act(self, _):
        return self.lr

    def load(self):
        return str(dataclasses.asdict(self))

    @classmethod
    def save(cls, pi_str):
        return cls(**literal_eval(pi_str))

    def reset(self, _):
        pass


@dataclasses.dataclass
class CosineAnnealingLRPolicy(AbstractPolicy):
    lr: float
    steps: int = dataclasses.field(init=False)

    def __post_init__(self):
        assert self.lr > 0

    def act(self, state):
        return 0.5 * (1 + np.cos(state["step"] * np.pi / self.steps)) * self.lr

    def load(self):
        return str(dataclasses.asdict(self))

    @classmethod
    def save(cls, pi_str):
        return cls(**literal_eval(pi_str))

    def reset(self, instance: generators.Instance):
        self.steps = instance.steps
        assert self.steps > 0


@dataclasses.dataclass
class SimplePolicy(AbstractPolicy):
    lr: float
    a: float
    b: float
    steps: int = dataclasses.field(init=False)
    loss: float = dataclasses.field(init=False)
    prev_loss: Optional[float] = dataclasses.field(init=False)
    epoch_size: int = dataclasses.field(init=False)

    def __post_init__(self):
        assert self.lr > 0
        assert 0 < self.a <= 1
        assert 0 < self.b <= 1

    def act(self, state):
        self.loss += state['loss'].sum()
        if not (state['step'] + 1) % self.epoch_size:
            if self.prev_loss is not None:
                self.lr *= self.a if self.loss > self.prev_loss else 1/self.b
            self.prev_loss = self.loss
            self.loss = 0.0
        return self.lr

    def load(self):
        return str(dataclasses.asdict(self))

    @classmethod
    def save(cls, pi_str):
        return cls(**literal_eval(pi_str))

    def reset(self, instance: generators.Instance):
        self.steps = instance.steps
        self.loss = 0.0
        self.prev_loss = None
        self.epoch_size = len(instance.loaders[0])
