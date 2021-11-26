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


# TODO: Seperate optimizer from instance
# TODO: Configurable controllable parameters
# TODO: Make seed part of the instance

class SGDEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, config=default_config):
        config = default_config.asdict()
        self.g = torch.Generator(device='cpu')
        self.instance_gen = self._create_instance_generator(**config.generator)

    def _create_instance_generator(self, generator_func, **kwargs):
        return generator_func(self.g, **kwargs)

    def step(self, action):
        for g in self.optimizer.param_groups:
            g['lr'] = action
        loss = self.epoch()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        step = self.optimizer.state[self.optimizer.param_groups[0]["params"][-1]]["step"]
        print(step)
        if step >= self.epochs:
            done = True
        else:
            done = False
        print(loss)
        return 1, 1, done, {}

    def reset(self):
        instance = next(self.instance_gen)
        self.model, self.optimizer, self.loss, loaders, self.epochs = instance
        self.train_loader, self.test_loader = loaders

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        if seed is not None:
            self.g.manual_seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        return [seed]

    def epoch(self):
        self.model.train()
        (data, target) = self.train_loader.next()
        output = self.model(data)
        loss = self.loss(output, target)
        return loss
