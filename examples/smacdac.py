import os
import argparse

import gym
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.facade.smac_ac_facade import SMAC4AC
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.scenario.scenario import Scenario

import sgd_env  # noqa
from examples.utils import SchedulerPolicyAction, run_policy


parser = argparse.ArgumentParser(
    description="Run SMAC to optimize initial learning rate of a scheduler"
)
parser.add_argument(
    "--policy",
    type=str,
    action=SchedulerPolicyAction,
    choices=SchedulerPolicyAction.names,
    required=True,
    help="Scheduler type",
)
parser.add_argument("--n_instances", type=int, default=1000, help="Number of instances")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()


env = gym.make("sgd-v0", n_instances=args.n_instances)
env.seed(args.seed)


def evaluate_cost(cfg, seed, instance, **kwargs):
    p = args.policy(**cfg)
    states, rewards = run_policy(env, p, int(instance))
    return -(rewards[-1] if states[-1]["crashed"] else np.sum(rewards))


def main():
    cs = ConfigurationSpace()
    cs.seed(args.seed)
    cs.add_hyperparameter(
        UniformFloatHyperparameter("lr", lower=0.000001, upper=10, log=True)
    )

    scenario = Scenario(
        {
            "run_obj": "quality",
            "runcount-limit": 5000,
            "cs": cs,
            "deterministic": True,
            "instances": [[i] for i in range(args.n_instances)],
            "output_dir": os.path.join("results", "smac"),
            "limit_resources": False,
            "abort_on_first_run_crash": False,
        }
    )

    smac = SMAC4AC(
        scenario=scenario,
        rng=args.seed,
        tae_runner=evaluate_cost,
        initial_design=RandomConfigurations,
    )

    incumbent = smac.optimize()

    incumbent_policy = args.policy.policy(**incumbent)
    with open(
        f"trained_{args.policy.__name__}_{args.seed}_{args.n_instances}", "w"
    ) as fh:
        incumbent_policy.save(fh)


if __name__ == "__main__":
    main()
