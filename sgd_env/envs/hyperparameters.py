import numpy as np
from dataclasses import dataclass


class Hyperparameter:
    pass


class UniformFloatHyperparameter(Hyperparameter):
    def __init__(self, low, high, log=False):
        self.low = low
        self.high = high
        self.log = log

    def sample(self, rng, size=None):
        low, high = self.low, self.high
        if self.log:
            low = np.log(low)
            high = np.log(high)
        value = rng.uniform(low, high, size=size)
        return np.exp(value) if self.log else value


class UniformIntegerHyperparameter(UniformFloatHyperparameter):
    def sample(self, rng, size=None):
        value = super().sample(rng, size)
        return np.rint(value)
