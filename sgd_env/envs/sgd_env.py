from itertools import cycle
from typing import Optional, Union, Iterator
import types

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import numpy as np
import torch

from .config import default_config
from . import utils


# TODO: GPU Support
# TODO: Inf sequence?


class SGDEnv(gym.Env, EzPickle):
    def __init__(self, config=default_config):
        self.config = default_config.asdict()
        self.instance_func = utils.create_instance_func(**self.config.generator)
        self.instance = cycle(range(self.config.dac.n_instances))
        self.np_random, seed = seeding.np_random()

        actions = {}
        for name, space, _ in self.config.dac.actions:
            actions[name] = space
        self.action_space = spaces.Dict(actions)
        self.instance_seeds = []

    def step(self, action):
        for name, _, func in self.config.dac.actions:
            func(self.optimizer, name, action)
        default_rng_state = torch.get_rng_state()
        torch.set_rng_state(self.env_rng_state)
        loss = utils.train(self.model, self.optimizer, self.loss, self.train_loader, 1)
        self._step += 1
        done = self._step >= self.steps
        # test_loss = self.test()
        self.env_rng_state = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        return loss.item(), -loss.item(), done, {}

    def reset(self, instance: Optional[Union[int, Iterator[int]]] = None):
        self._step = 0
        if len(self.instance_seeds) == 0:
            rng = np.random.RandomState()
            self.instance_seeds = rng.randint(
                1, 4294967295, self.config.dac.n_instances
            )

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
        self.model, optimizer_params, self.loss, loaders, self.steps = instance
        self.train_loader, self.test_loader = loaders
        self.optimizer = utils.create_optimizer(
            **self.config.optimizer, **optimizer_params, params=self.model.parameters()
        )
        self.model.eval()
        (data, target) = self.train_loader.next()
        output = self.model(data)
        loss = self.loss(output, target)
        self.env_rng_state = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        return loss.item()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        if seed is not None:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.instance_seeds = self.np_random.randint(
            1, 4294967295, self.config.dac.n_instances
        )
        return [seed]

    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += self.loss(output, target, reduction="sum").item()
        test_loss /= len(self.test_loader.dataset)
        return test_loss
