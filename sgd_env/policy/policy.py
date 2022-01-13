import dataclasses
from abc import ABC, abstractmethod
from ast import literal_eval

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

    def act(self, state):
        return 0.5 * (1 + np.cos(state["step"] * np.pi / self.steps)) * self.lr

    def load(self):
        return str(dataclasses.asdict(self))

    @classmethod
    def save(cls, pi_str):
        return cls(**literal_eval(pi_str))

    def reset(self, instance: generators.Instance):
        self.steps = instance.steps
