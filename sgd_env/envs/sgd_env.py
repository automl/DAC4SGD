from itertools import cycle, count
from typing import Optional, Union, Iterator, Tuple

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import numpy as np
import torch

from sgd_env.envs.generators import Instance, default_instance_generator, GeneratorFunc
from sgd_env.envs import utils


class SGDEnv(gym.Env, EzPickle):
    def __init__(
        self,
        generator: GeneratorFunc = default_instance_generator,
        n_instances: Union[int, float] = np.inf,
        device: str = "cpu",
    ):
        self.generator = generator
        self.n_instances = n_instances
        self.device = device
        self.seed()

        self.action_space = spaces.Box(low=0.0, high=np.inf, shape=(1,))
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

    def step(self, action: float):
        utils.optimizer_action(self.optimizer, "lr", {"lr": action})
        default_rng_state = torch.get_rng_state()
        torch.set_rng_state(self.env_rng_state)
        train_args = [
            self.model,
            self.optimizer,
            self.loss,
            self.train_iter,
            1,
            self.device,
        ]
        try:
            loss = utils.train(*train_args)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            loss = utils.train(*train_args)
        self._step += 1
        self.env_rng_state.data = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        crashed = (
            not torch.isfinite(loss).any()
            or not torch.isfinite(
                torch.nn.utils.parameters_to_vector(self.model.parameters())
            ).any()
        )
        state = {"step": self._step, "loss": loss, "crashed": crashed}
        done = self._step >= self.cutoff if not crashed else True
        reward = -loss.mean().item() if not crashed else self.crash_penalty
        return state, reward, done, {}

    def reset(self, instance: Optional[Union[Instance, int]] = None):
        self._step = 0
        default_rng_state = torch.get_rng_state()

        if isinstance(instance, Instance):
            self.instance = instance
            (
                self.dataset,
                self.model,
                optimizer_params,
                self.loss,
                self.batch_size,
                (self.train_loader, _),
                self.cutoff,
                self.crash_penalty,
            ) = instance
        else:
            if instance is None:
                instance_idx = next(self.instance_count)
            elif isinstance(instance, int):
                instance_idx = instance
            else:
                raise NotImplementedError

            assert instance_idx < self.n_instances
            while instance_idx >= len(self.instance_seeds):
                seed = self.np_random.randint(1, 4294967295, dtype=np.int64)
                self.instance_seeds.append(seed)

            seed = self.instance_seeds[instance_idx]
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            rng = np.random.RandomState(seed)

            self.instance = self.generator(rng)
            assert isinstance(self.instance, Instance)
            (
                self.dataset,
                self.model,
                optimizer_params,
                self.loss,
                self.batch_size,
                (self.train_loader, _),
                self.cutoff,
                self.crash_penalty,
            ) = self.instance

        self._observation_space = spaces.Dict(
            {
                "step": spaces.Box(0, self.cutoff, (1,)),
                "loss": spaces.Box(0, np.inf, (self.batch_size,)),
                "crashed": spaces.Discrete(1),
            }
        )

        self.train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]] = iter(
            self.train_loader
        )
        self.model.to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            **optimizer_params, params=self.model.parameters()
        )
        self.model.eval()
        (data, target) = next(self.train_iter)
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = self.loss(output, target, reduction="none")
        self.env_rng_state: torch.Tensor = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        return {"step": 0, "loss": loss, "crashed": False}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.instance_seeds = []
        if self.n_instances == np.inf:
            self.instance_count = count(start=0, step=1)
        else:
            self.instance_count = cycle(range(self.n_instances))
        return [seed]
