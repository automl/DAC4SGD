import operator
import functools
import random
import inspect
from itertools import cycle
from typing import Optional, Union, Iterator
import types

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import utils

import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F

from .generators import random_instance
from .config import default_config


# TODO: GPU Support
# TODO: Custom action {'name': 'reset', 'type': spaces.Binary, 'func': func}
#       func(optimizer, actions) -> None change optimizer state directly


MAX_SEED = 4025501053080439804
NP_MAX_SEED = 4294967295


def random_seeds(rng, n):
    seeds = []
    for _ in range(n):
        seeds.append(rng.randint(1, NP_MAX_SEED))
    return seeds


class SGDEnv(gym.Env, utils.EzPickle):
    def __init__(self, config=default_config):
        self.config = default_config.asdict()
        self.instance_func = self._create_instance_func(**self.config.generator)
        self.instance = cycle(range(self.config.dac.n_instances))
        self.np_random, seed = seeding.np_random()

        actions = {}
        sig = inspect.signature(self.config.optimizer.optimizer)
        for name in self.config.dac.control:
            if name in sig.parameters:
                value = sig.parameters[name]
                if isinstance(value, bool):
                    action = spaces.Binary()
                else:
                    action = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
                actions[name] = action
        self.action_space = spaces.Dict(actions)
        self.instance_seeds = []

    def _create_instance_func(self, generator_func, **kwargs):
        return functools.partial(generator_func, **kwargs)

    def create_optimizer(self, optimizer, params, **kwargs):
        return optimizer(params, **kwargs)

    def step(self, action):
        for g in self.optimizer.param_groups:
            g["lr"] = action["lr"]
        default_rng_state = torch.get_rng_state()
        torch.set_rng_state(self.env_rng_state)
        loss = self.epoch()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._step += 1
        done = self._step >= self.epochs
        # test_loss = self.test()
        self.env_rng_state = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        return 1, -loss.item(), done, {}

    def reset(self, instance: Optional[Union[int, Iterator[int]]] = None):
        self._step = 0
        if len(self.instance_seeds) == 0:
            rng = np.random.RandomState()
            self.instance_seeds = random_seeds(rng, self.config.dac.n_instances)

        if instance is None:
            instance = next(self.instance)
        elif isinstance(instance, types.GeneratorType):
            instance = next(instance)
        elif not isinstance(instance, int):
            raise NotImplementedError

        default_rng_state = torch.get_rng_state()
        seed = self.instance_seeds[instance]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.RandomState(seed)

        instance = self.instance_func(rng)
        self.model, optimizer_params, self.loss, loaders, self.epochs = instance
        self.train_loader, self.test_loader = loaders
        self.optimizer = self.create_optimizer(
            **self.config.optimizer, **optimizer_params, params=self.model.parameters()
        )
        self.env_rng_state = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        if seed is not None:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.instance_seeds = random_seeds(self.np_random, self.config.dac.n_instances)
        return [seed]

    def epoch(self):
        self.model.train()
        (data, target) = self.train_loader.next()
        output = self.model(data)
        loss = self.loss(output, target)
        return loss

    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += self.loss(output, target, reduction="sum").item()
        test_loss /= len(self.test_loader.dataset)
        return test_loss
