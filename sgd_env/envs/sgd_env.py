from itertools import cycle, count
from typing import Optional, Union, Iterator
import types

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import numpy as np
import torch

from .config import default_config
from . import utils


class SGDEnv(gym.Env, EzPickle):
    def __init__(self, config=default_config):
        self.config = default_config.asdict()
        self.instance_func = utils.create_instance_func(**self.config.generator)
        self.seed(self.config.dac.seed)

        actions = {name: space for name, space, _ in self.config.dac.actions}
        self.action_space = spaces.Dict(actions)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,))

    def step(self, action):
        for name, _, func in self.config.dac.actions:
            func(self.optimizer, name, action)
        default_rng_state = torch.get_rng_state()
        torch.set_rng_state(self.env_rng_state)
        loss = utils.train(
            self.model,
            self.optimizer,
            self.loss,
            self.train_loader,
            1,
            self.config.dac.device,
        )
        self._step += 1
        done = self._step >= self.steps
        self.env_rng_state = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        return loss.item(), -loss.item(), done, {}

    def reset(self, instance: Optional[Union[int, Iterator[int]]] = None):
        self._step = 0

        if instance is None:
            instance = next(self.instance)
        elif isinstance(instance, types.GeneratorType):
            instance = next(instance)
        elif not isinstance(instance, int):
            raise NotImplementedError

        default_rng_state = torch.get_rng_state()
        assert instance < self.config.dac.n_instances
        if instance <= len(self.instance_seeds):
            seed = self.np_random.randint(1, 4294967295)
            self.instance_seeds.append(seed)

        seed = self.instance_seeds[instance]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.RandomState(seed)

        (
            self.model,
            optimizer_params,
            self.loss,
            (self.train_loader, self.test_loader),
            self.steps,
        ) = self.instance_func(rng)
        self.model.to(self.config.dac.device)
        self.optimizer = utils.create_optimizer(
            **self.config.optimizer, **optimizer_params, params=self.model.parameters()
        )
        self.model.eval()
        (data, target) = self.train_loader.next()
        data, target = data.to(self.config.dac.device), target.to(
            self.config.dac.device
        )
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
        self.instance_seeds = []
        if self.config.dac.n_instances == np.inf:
            self.instance = count(start=0, step=1)
        else:
            self.instance = cycle(range(self.config.dac.n_instances))
        return [seed]
