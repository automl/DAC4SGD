import operator
import functools
import random
import inspect
from typing import Optional, Union, Iterator

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import utils

import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F

from .generators import random_instance_generator
from .config import default_config


# TODO: GPU Support
# TODO: Convert torch.random to np.RandomState for instance selection
# TODO: Implement instance selection to reset (Generator or int)
# TODO: Custom action {'name': 'reset', 'type': spaces.Binary, 'func': func}
#       func(optimizer, actions) -> None change optimizer state directly
# TODO: Change default architecture generator to have fixed depth, structure


MAX_SEED = 4025501053080439804
NP_MAX_SEED = 4294967295


def random_states(rng, n):
    default_state = np.random.get_state()
    states = []
    for _ in range(n):
        np.random.seed(rng.randint(1, NP_MAX_SEED))
        state = np.random.get_state()
        states.append(state)
    np.random.set_state(default_state)
    return states


class SGDEnv(gym.Env, utils.EzPickle):
    def __init__(self, config=default_config):
        self.config = default_config.asdict()
        self.g = torch.Generator(device='cpu')
        self.instance_gen = self._create_instance_generator(**self.config.generator)

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
        self.instance_random_states = []


    def _create_instance_generator(self, generator_func, **kwargs):
        return generator_func(self.g, **kwargs)

    def create_optimizer(self, optimizer, params, **kwargs):
        return optimizer(params, **kwargs)

    def step(self, action):
        for g in self.optimizer.param_groups:
            g['lr'] = action['lr']
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
        if len(self.instance_random_states) == 0:
            rng = np.random.default_rng()
            self.instance_random_states = random_states(rng, self.config.dac.n_instances)

        default_rng_state = torch.get_rng_state()
        seed = torch.randint(0, MAX_SEED, (), generator=self.g)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        instance = next(self.instance_gen)
        self.model, optimizer_params, self.loss, loaders, self.epochs = instance
        self.train_loader, self.test_loader = loaders
        self.optimizer = self.create_optimizer(
            **self.config.optimizer,
            **optimizer_params,
            params=self.model.parameters())
        self.env_rng_state = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        if seed is not None:
            self.g.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.instance_random_states = random_states(self.np_random, self.config.dac.n_instances)
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
                test_loss += self.loss(output, target, reduction='sum').item()
        print(len(self.test_loader.dataset))
        test_loss /= len(self.test_loader.dataset)
        return test_loss
