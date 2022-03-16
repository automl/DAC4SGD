import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Union

import stable_baselines3
from dac4automlcomp.policy import DACPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from utils import convert_observation, convert_action


class SGDPolicy(DACPolicy):
    """
    Policy for adjusting the learning rate based on the state (step, loss and crashed) of
    the SGD learning.

    Intended to work with original SGDEnv. This is why we need to convert observations
    and actions in self.act to the format required by the original env.
    """

    def __init__(
        self,
        agent: BaseAlgorithm,
        args: Optional[argparse.Namespace] = None,
        deterministic: bool = False,
    ):
        self.agent = agent
        self.args = args
        self.deterministic = deterministic

    def act(self, state: Dict):
        """
        Predict the next action depending on state.

        Because our policy requires the state to be a different format than in the original
        environment, we convert the observations.
        Likewise, we predict the log-learning rate but the environment expects the
        normal learning rate, so we convert the action as well.

        Parameters
        ----------
        state: Dict
            Observation coming from SGDEnv.
        Returns
        -------
        action
            Normal learning rate.
        """
        state = convert_observation(observation=state)
        # next_state will be None because we don't use an RNN
        action, next_state = self.agent.predict(
            observation=state, deterministic=self.deterministic
        )
        action = convert_action(action=action)
        return action

    def save(self, agent_fn: Union[str, Path]):
        agent_fn = Path(agent_fn)
        agent_fn.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(path=agent_fn)

    def reset(self, instance):
        pass

    def seed(self, seed):
        pass

    @classmethod
    def load(cls, model_fn: Union[str, Path]):
        model_fn = Path(model_fn)
        logdir = model_fn.parent

        # Load arguments
        args_fn = logdir / "args.json"
        with open(args_fn, "r") as file:
            args = json.load(file)
        args = argparse.Namespace(**args)

        # Load agent
        model_fn = logdir / "model.zip"
        agent = load_agent(agent=args.agent, model_fn=model_fn)

        sgd_policy = cls(agent=agent, args=args)

        return sgd_policy


def load_agent(agent: str, model_fn: Union[str, Path]) -> BaseAlgorithm:
    agent_cls = getattr(stable_baselines3, agent)
    agent = agent_cls.load(str(model_fn))

    return agent
