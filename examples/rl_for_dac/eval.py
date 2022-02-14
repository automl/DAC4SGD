import json
import argparse
from typing import Union
from pathlib import Path
import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm

from sgd_env.policy.policy import AbstractPolicy
from sgd_env.envs import generators


def load_agent(agent: str, model_fn: Union[str, Path]) -> BaseAlgorithm:
    agent_cls = getattr(stable_baselines3, agent)
    agent = agent_cls.load(str(model_fn))

    return agent


class SGDPolicy(AbstractPolicy):
    def __init__(self, agent: BaseAlgorithm, deterministic: bool = False):
        self.agent = agent
        self.deterministic = deterministic

    def act(self, state):
        return self.agent.predict(observation=state, deterministic=self.deterministic)

    def reset(self, instance: generators.Instance):
        # We don't use instance features here so pass
        pass

    def save(self, agent_fn: Union[str, Path]):
        agent_fn = Path(agent_fn)
        agent_fn.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(path=agent_fn)

    @classmethod
    def load(cls, model_fn: Union[str, Path]):
        model_fn = Path(model_fn)
        logdir = model_fn.parent

        # Load arguments
        args_fn = logdir / "args.json"
        with open(args_fn, 'r') as file:
            args = json.load(file)
        args = argparse.Namespace(**args)

        # Load agent
        model_fn = logdir / "model.zip"
        agent = load_agent(agent=args.agent, model_fn=model_fn)

        sgd_policy = cls(
            agent=agent,
        )

        return sgd_policy


if __name__ == '__main__':
    model_fn = "tmp/model.zip"
    sgd_policy = SGDPolicy.load(model_fn)


