from typing import Iterator, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces

from dac4automlcomp.dac_env import DACEnv
from dac4automlcomp.generator import Generator
from sgd_env.envs import utils
from sgd_env.envs.generators import DefaultSGDGenerator, SGDInstance


class SGDEnv(DACEnv[SGDInstance]):
    def __init__(
        self,
        generator: Generator[SGDInstance] = DefaultSGDGenerator(),
        n_instances: Union[int, float] = np.inf,
        device: str = "cpu",
    ):
        super().__init__(generator, n_instances)
        self.device = device

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

    @DACEnv._get_instance.register
    def _(self, instance: SGDInstance):
        return instance

    def step(self, action: float):
        action = float(action)  # convert to float if we receive a tensor
        utils.optimizer_action(self.optimizer, "lr", {"lr": action})
        default_rng_state = torch.get_rng_state()
        torch.set_rng_state(self.env_rng_state)
        train_args = [
            self.model,
            self.optimizer,
            self.loss_function,
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
        self._step += 1
        self.env_rng_state.data = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        crashed = (
            not torch.isfinite(loss).any()
            or not torch.isfinite(
                torch.nn.utils.parameters_to_vector(self.model.parameters())
            ).any()
        )
        validation_loss = None
        if self._step % len(self.train_loader) == 0:
            validation_loss = utils.test(
                self.model, self.loss_function, self.validation_loader, self.device
            )
        state = {
            "step": self._step,
            "loss": loss,
            "validation_loss": validation_loss,
            "crashed": crashed,
        }
        done = self._step >= self.cutoff if not crashed else True
        if crashed:
            reward = self.crash_penalty
        elif done:
            test_losses = utils.test(
                self.model, self.loss_function, self.test_loader, self.device
            )
            reward = -test_losses.sum() / len(self.test_loader.dataset)
        else:
            reward = 0.0
        return state, reward, done, {}

    def reset(self, instance: Optional[Union[SGDInstance, int]] = None):
        super().reset(instance)
        self._step = 0
        default_rng_state = torch.get_rng_state()

        assert isinstance(self.current_instance, SGDInstance)
        (
            self.dataset,
            self.model,
            optimizer_params,
            self.loss_function,
            self.batch_size,
            self.train_validation_ratio,
            (self.train_loader, self.validation_loader, self.test_loader),
            self.cutoff,
            self.crash_penalty,
        ) = self.current_instance

        self._observation_space = spaces.Dict(
            {
                "step": spaces.Box(0, self.cutoff, (1,)),
                "loss": spaces.Box(0, np.inf, (self.batch_size,)),
                "validation_loss": spaces.Box(
                    0, np.inf, (len(self.test_loader.dataset),)
                ),
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
        loss = self.loss_function(output, target, reduction="none")
        self.env_rng_state: torch.Tensor = torch.get_rng_state()
        torch.set_rng_state(default_rng_state)
        return {
            "step": 0,
            "loss": loss,
            "validation_loss": None,
            "crashed": False,
        }

    def seed(self, seed=None):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return super().seed(seed)
