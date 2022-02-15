import json
import argparse
from typing import Union, Optional
from pathlib import Path
import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm

from sgd_env.policy.policy import AbstractPolicy
from sgd_env.envs import generators

from examples.rl_for_dac.train import make_sgd_env
from examples.utils import run_policy


def load_agent(agent: str, model_fn: Union[str, Path]) -> BaseAlgorithm:
    agent_cls = getattr(stable_baselines3, agent)
    agent = agent_cls.load(str(model_fn))

    return agent


class SGDPolicy(AbstractPolicy):
    def __init__(self, agent: BaseAlgorithm, args: Optional[argparse.Namespace] = None, deterministic: bool = False):
        self.agent = agent
        self.args = args
        self.deterministic = deterministic

    def act(self, state):
        # next_state will be None because we don't use an RNN
        action, next_state = self.agent.predict(observation=state, deterministic=self.deterministic)
        return action

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
            args=args
        )

        return sgd_policy


if __name__ == '__main__':
    import numpy as np

    # Load model and env
    model_fn = "tmp/model.zip"
    sgd_policy = SGDPolicy.load(model_fn)
    env = make_sgd_env(args=sgd_policy.args)

    # Exemplarily evaluate
    states, rewards = run_policy(env=env, policy=sgd_policy, instance=None)
    cumulative_reward = np.sum(rewards)
    print("Cumulative Reward: ", cumulative_reward)


