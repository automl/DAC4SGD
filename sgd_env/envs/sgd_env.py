from itertools import count, cycle
from typing import Iterator, Optional, Tuple, Union, Dict

import gym
import numpy as np
import torch
from gym import spaces
from dac4automlcomp.dac_env import DACEnv, Instance, Generator

from sgd_env.envs import utils
from sgd_env.envs.generators import default_instance_generator


class SGDEnv(DACEnv):
    def __init__(
        self,
        generator: Generator = default_instance_generator,
        instance_set: Dict = None,
        n_instances: Union[int, float] = np.inf,
        device: str = "cpu",
    ):
        super().__init__(generator=generator, instance_set=instance_set, n_instances=n_instances, device=device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if self.n_instances == np.inf:
            self.instance_count = count(start=0, step=1)
        else:
            self.instance_count = cycle(range(self.n_instances))
        self.action_space = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        self._observation_space = None
        self.train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]]

    @property
    def observation_space(self):
        if self._observation_space is None:
            raise ValueError(
                "Observation space changes for every instance. "
                "It is set after every reset. "
                "Use a provided wrapper or handle it manually. "
                "If batch size is fixed for every instance, "
                "observation space will stay fixed."
            )
        return self._observation_space

    def step(self, action: float):
        done = self._step()
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
            train_args[3] = self.train_iter
            loss = utils.train(*train_args)
        self.env_rng_state.data = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        crashed = (
            not torch.isfinite(loss).any()
            or not torch.isfinite(
                torch.nn.utils.parameters_to_vector(self.model.parameters())
            ).any()
        )
        state = {"step": self._step, "loss": loss, "crashed": crashed}
        done = done if not crashed else True
        if crashed:
            reward = self.crash_penalty
        elif done:
            reward = -utils.test(
                self.model, self.loss, self.validation_loader, self.device
            )
        else:
            reward = 0.0
        return state, reward, done, {}

    def reset(self, instance: Optional[Union[Instance, int]] = None):
        seed = self._reset(instance)
        default_rng_state = torch.get_rng_state()
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        (
            self.dataset,
            self.model,
            optimizer_params,
            self.loss,
            self.batch_size,
            (self.train_loader, self.validation_loader),
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

        self.train_iter = iter(self.train_loader)
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
