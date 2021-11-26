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

from .dataclass_config import Config

new_mirror = "https://ossci-datasets.s3.amazonaws.com/mnist"
datasets.MNIST.resources = [
    ("/".join([new_mirror, url.split("/")[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
]

default_config = Config()


def random_feature_extractor(rng):
    n_layers = torch.randint(low=2, high=5, size=(), generator=rng)
    use_dropout = torch.randint(high=2, size=(), generator=rng)
    use_pooling = torch.randint(high=2, size=(), generator=rng)
    in_size = 1
    layers = []
    for i in range(n_layers):
        out_size = torch.randint(low=16, high=129, size=(),
                generator=rng)
        layers.append(nn.Conv2d(in_size, out_size, 3, 1))
        if use_dropout:
            layers.append(nn.Dropout(torch.rand(size=(), generator=rng)))
        layers.append(nn.ReLU())
        in_size = out_size

    if use_pooling:
        layers.append(nn.MaxPool2d(torch.randint(low=2, high=5,
            size=(), generator=rng).item()))

    return nn.Sequential(*layers)


def random_mnist_model(rng):
    class MNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = random_feature_extractor(rng)
            n_features = torch.prod(torch.tensor(self.f(torch.rand((1, 1, 28,
                28), generator=rng)).shape))
            size = torch.randint(low=16, high=129, size=())
            self.fc1 = nn.Linear(n_features, size)
            self.fc2 = nn.Linear(size, 10)

        def forward(self, x):
            x = self.f(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    return MNISTModel()


def random_mnist_loader(rng):
      transform=transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
          ])
      train_kwargs = {'batch_size': 32}
      test_kwargs = {'batch_size': 32}
      dataset1 = datasets.MNIST('data', train=True, download=True,
                         transform=transform)
      dataset2 = datasets.MNIST('data', train=False,
                         transform=transform)
      train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
      test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
      return iter(train_loader), iter(test_loader)


def random_mnist_instance(rng):
    model = random_mnist_model(rng)
    loaders = random_mnist_loader(rng)
    loss = nn.NLLLoss()
    epochs = torch.randint(low=300, high=900, size=(), generator=rng)
    log_lr = (np.log(0.0001) - np.log(1)) * torch.rand((), generator=rng) + np.log(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=np.exp(log_lr))
    return model, optimizer, loss, loaders, epochs


def random_instance_generator(rng):
    datasets = ['MNIST']
    while True:
        idx = torch.randint(high=len(datasets), size=(), generator=rng)
        dataset = datasets[idx.item()]
        if dataset == 'MNIST':
            instance = random_mnist_instance(rng)
        else:
            raise NotImplementedError
        yield instance


class SGDEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None):
        self.g = torch.Generator(device='cpu')
        self.instance_gen = random_instance_generator(self.g)

    def step(self, action):
        for g in self.optimizer.param_groups:
            g['lr'] = action
        loss = self.epoch()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        step = self.optimizer.state[self.optimizer.param_groups[0]["params"][-1]]["step"]
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
