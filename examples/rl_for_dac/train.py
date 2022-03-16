"""
Reinforcement Learning for SGD Env.

Model as a contextual MDP [Hallak et al., 2015, Biedenkapp et al., 2020].

Action: continous, learning rate. log-space. 0.00001 to 1
State: step, loss, crashed
Reward: test loss
Instance Schedule: Round Robin


"""
import argparse
from pathlib import Path
from rich import print
import numpy as np
import torch as th
import json
from typing import List, Dict
import datetime

import gym
from gym import ObservationWrapper, ActionWrapper
import stable_baselines3

import sgd_env  # noqa

from .utils import CustomCheckpointCallback, convert_observation, convert_action


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train a RL agent to dynamically adjust learning rate."
    )
    parser.add_argument(
        "--n_instances", type=int, default=1000, help="Number of instances"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agent", type=str, default="SAC", help="RL agent")
    parser.add_argument(
        "--net_arch",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Network architecture, List of layers with number of hidden units.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1e3,
        help="Number of total timesteps that should be "
        "taken in the environment for training.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="tmp",
        help="Directory where to save trained models and logs.",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=5000,
        help="Save frequency of model (number of timesteps).",
    )
    return parser


class SGDEnvObservationWrapper(ObservationWrapper):
    """
    SGDEnv returns the observations as a dictionary. Return a vector.
    """

    def observation(self, observation: Dict) -> th.Tensor:
        return convert_observation(observation=observation)


class SGDEnvActionWrapper(ActionWrapper):
    def action(self, action):
        return convert_action(action=action)


def make_sgd_env(args):
    env = gym.make("sgd-v0", n_instances=args.n_instances)
    env.seed(args.seed)

    # set observation space
    low = np.array([0, 0, 0])  # step, loss, crashed
    high = np.array([np.inf, np.inf, 1])
    env._observation_space = gym.spaces.Box(low=low, high=high, shape=(3,))

    # Wrap the observations (Dict to Vector)
    env = SGDEnvObservationWrapper(env=env)

    # Wrap actions (numpy to torch)
    env = SGDEnvActionWrapper(env=env)

    # set action space
    low = np.log(1e-5)
    high = np.log(1)
    env.action_space = gym.spaces.Box(low=low, high=high, shape=(1,))  # log space

    return env


def main(
    args: argparse.Namespace, unknown_args: List[str], parser: argparse.ArgumentParser
):
    print(args)

    # Setup logging
    run_id = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    logdir = Path(args.outdir) / run_id
    logdir.mkdir(parents=True, exist_ok=True)
    logger = stable_baselines3.common.logger.configure(
        str(logdir), ["stdout", "csv", "tensorboard"]
    )

    # Save args
    kwargs = args._get_kwargs()
    kwargs_dict = {k: v for (k, v) in kwargs}
    args_fn = logdir / "args.json"
    with open(args_fn, "w") as file:
        json.dump(kwargs_dict, file, indent="\t")

    # Make environment
    env = make_sgd_env(args=args)

    # Select agent
    agent_cls = getattr(stable_baselines3, args.agent)
    # Select policy
    agent_kwargs = {
        "env": env,
        "seed": args.seed,
        "policy": "MlpPolicy",
        "policy_kwargs": {"net_arch": args.net_arch},
        "verbose": 1,
    }
    # Create agent
    agent = agent_cls(**agent_kwargs)
    agent.set_logger(logger)

    # Checkpointing callback
    chkp_cb = CustomCheckpointCallback(
        save_freq=args.save_freq,
        name_prefix="model",
        save_path=str(logdir),
        override=True,
    )
    callbacks = [chkp_cb]

    # Learn
    agent.learn(total_timesteps=args.n_steps, callback=callbacks)

    # Save model
    savepath = logdir / "model.zip"
    agent.save(path=savepath)

    return savepath


if __name__ == "__main__":
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    model_savepath = main(args, unknown_args, parser)
