from abc import ABC, abstractmethod
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
