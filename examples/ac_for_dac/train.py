import argparse
from pathlib import Path

import gym
from dac4automlcomp.policy import DeterministicPolicy
from smac.facade.smac_ac_facade import SMAC4AC
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.scenario.scenario import Scenario

from examples.ac_for_dac import schedulers


class SchedulerPolicyAction(argparse.Action):
    names = [
        cls.__name__ for cls in schedulers.Configurable.__subclasses__()
    ]

    def __call__(self, parser, namespace, values, option_string=None):
        mapping = dict(
            zip(self.names, schedulers.Configurable.__subclasses__())
        )
        setattr(namespace, self.dest, mapping[values])


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run SMAC to optimize the hyperparameters of a hand-crafted learning rate scheduler"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=schedulers.ConstantLRPolicy,
        action=SchedulerPolicyAction,
        choices=SchedulerPolicyAction.names,
        help="Scheduler class name",
    )
    parser.add_argument("--n_instances", type=int, default=1000, help="Number of instances in training environment")
    parser.add_argument("--env_seed", type=int, default=42, help="Random seed for the training environment")
    parser.add_argument("--smac_seed", type=int, default=42, help="Random seed for SMAC")
    parser.add_argument("--smac_budget", type=int, default=5000, help="Budget for SMAC (# TAE calls)")
    parser.add_argument("--outdir", type=str, default="tmp", help="Directory where to save trained models and logs.")
    return parser


def evaluate_cost(cfg, seed, instance, **kwargs):
    global args
    global train_env
    policy = args.policy.from_config(cfg)
    policy.seed(seed)
    obs = train_env.reset(instance=int(instance))
    policy.reset(train_env.current_instance)
    done = False
    total_reward = 0
    while not done:
        action = policy.act(obs)
        obs, reward, done, _ = train_env.step(action)
        total_reward += reward
    return -(train_env.current_instance.crash_penalty if obs["crashed"] else total_reward)


if __name__ == "__main__":
    global args
    global train_env

    args = get_parser().parse_args()

    logdir = Path(args.outdir)
    logdir.mkdir(parents=True, exist_ok=True)

    train_env = gym.make("sgd-v0", n_instances=args.n_instances)
    train_env.seed(args.env_seed)

    scenario = Scenario(
        {
            "run_obj": "quality",
            "runcount-limit": args.smac_budget,
            "cs": args.policy.config_space(),
            "deterministic": isinstance(args.policy, DeterministicPolicy),
            "instances": [[i] for i in range(args.n_instances)],
            "output_dir": logdir / "smac_logs" / args.policy.__name__,
            "limit_resources": False,
            "abort_on_first_run_crash": False,
        }
    )

    smac = SMAC4AC(
        scenario=scenario,
        rng=args.smac_seed,
        tae_runner=evaluate_cost,
        initial_design=RandomConfigurations,
    )

    incumbent = smac.optimize()

    incumbent_policy = args.policy.from_config(incumbent)
    config_save_dir = logdir / "saved_configs"
    config_save_dir.mkdir(parents=True, exist_ok=True)
    incumbent_policy.save(config_save_dir)
