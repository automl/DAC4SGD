from itertools import cycle, count
from functools import partial
from typing import Optional, Union, Iterator, Tuple, Protocol
import types

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import numpy as np
import torch

from .generators import Instance, random_instance
from .hyperparameters import UniformIntegerHyperparameter
from .hyperparameters import UniformFloatHyperparameter
from . import utils


class GeneratorFunc(Protocol):
    def __call__(self, rng: np.random.RandomState, **kwargs: int) -> Instance:
        ...


class SGDEnv(gym.Env, EzPickle):
    def __init__(
        self,
        generator: GeneratorFunc = partial(
            random_instance,
            steps=UniformIntegerHyperparameter(300, 900),
            learning_rate=UniformFloatHyperparameter(0.0001, 0.1, True),
            batch_size=UniformIntegerHyperparameter(32, 256, True),
        ),
        n_instances: Union[int, float] = np.inf,
        device: str = "cpu",
    ):
        self.generator = generator
        self.n_instances = n_instances
        self.device = device
        self.seed()

        self.actions = [
            (
                "lr",
                spaces.Box(low=0.0, high=np.inf, shape=(1,)),
                utils.optimizer_action,
            )
        ]

        self.action_space = spaces.Dict(
            {name: space for name, space, _ in self.actions}
        )
        self._observation_space = None

    @property
    def observation_space(self):
        if self._observation_space is None:
            raise ValueError(
                "Observation space changes for every instance. "
                "It is set after every reset. "
                "Use a provided wrapper or handle it manually."
                "If batch size is fixed for every instance, "
                "observation space will stay fixed."
            )
        return self._observation_space

    def step(self, action: spaces.Dict):
        for name, _, func in self.actions:
            func(self.optimizer, name, action)
        default_rng_state = torch.get_rng_state()
        torch.set_rng_state(self.env_rng_state)
        loss = utils.train(
            self.model,
            self.optimizer,
            self.loss,
            self.train_loader,
            1,
            self.device,
        )
        self._step += 1
        self.env_rng_state.data = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        crashed = (
            torch.isnan(loss).any()
            or torch.isnan(
                torch.nn.utils.parameters_to_vector(self.model.parameters())
            ).any()
        ).item()
        state = {"steps": self._step, "loss": loss, "crashed": crashed}
        done = self._step >= self.steps if not crashed else True
        reward = -loss.mean() if not crashed else self.crash_penalty
        return state, reward, done, {}

    def reset(self, instance: Optional[Union[Instance, int, Iterator[int]]] = None):
        self._step = 0
        default_rng_state = torch.get_rng_state()

        if isinstance(instance, Instance):
            (
                self.dataset,
                self.model,
                optimizer_params,
                self.loss,
                self.batch_size,
                (train_loader, _),
                self.steps,
                self.crash_penalty,
            ) = instance
        else:
            if instance is None:
                instance_idx = next(self.instance)
            elif isinstance(instance, types.GeneratorType):
                instance_idx = next(instance)
            elif isinstance(instance, int):
                instance_idx = instance
            else:
                raise NotImplementedError

            assert instance_idx < self.n_instances
            while instance_idx >= len(self.instance_seeds):
                seed = self.np_random.randint(1, 4294967295)
                self.instance_seeds.append(seed)

            seed = self.instance_seeds[instance_idx]
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            rng = np.random.RandomState(seed)

            (
                self.dataset,
                self.model,
                optimizer_params,
                self.loss,
                self.batch_size,
                (train_loader, _),
                self.steps,
                self.crash_penalty,
            ) = self.generator(rng)

        self._observation_space = spaces.Dict(
            {
                "steps": spaces.Box(0, self.steps, (1,)),
                "loss": spaces.Box(0, np.inf, (self.batch_size,)),
                "crashed": spaces.Discrete(1),
            }
        )

        self.train_loader: Iterator[Tuple[torch.Tensor, torch.Tensor]] = iter(
            train_loader
        )
        self.model.to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            **optimizer_params, params=self.model.parameters()
        )
        self.model.eval()
        (data, target) = next(self.train_loader)
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = self.loss(output, target, reduction="none")
        self.env_rng_state: torch.Tensor = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        return {"steps": 0, "loss": loss, "crashed": False}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.instance_seeds = []
        if self.n_instances == np.inf:
            self.instance = count(start=0, step=1)
        else:
            self.instance = cycle(range(self.n_instances))
        return [seed]
