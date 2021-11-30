import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import utils
import operator
import functools
import random

import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F

from .generators import random_instance_generator
from .config import default_config


# TODO: GPU Support
# TODO: Configurable controllable parameters

MAX_SEED = 4025501053080439804


class SGDEnv(gym.Env, utils.EzPickle):
    def __init__(self, config=default_config):
        self.config = default_config.asdict()
        self.g = torch.Generator(device='cpu')
        self.instance_gen = self._create_instance_generator(**self.config.generator)

    def _create_instance_generator(self, generator_func, **kwargs):
        return generator_func(self.g, **kwargs)

    def create_optimizer(self, optimizer, params, **kwargs):
        return optimizer(params, **kwargs)

    def step(self, action):
        for g in self.optimizer.param_groups:
            g['lr'] = action
        loss = self.epoch()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._step += 1
        done = self._step >= self.epochs
        # test_loss = self.test()
        return 1, -loss.item(), done, {}

    def reset(self):
        self._step = 0

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        if seed is not None:
            self.g.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
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
