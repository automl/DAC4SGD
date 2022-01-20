"""Example of using SMAC 3 (static algorithm configurator) for DAC by configuring a parameterized DAC policy rather
than the target algorithm.

Note: This code does not work on Windows (SMAC 3 does not support Windows)
"""
import traceback

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.facade.smac_ac_facade import SMAC4AC
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.scenario.scenario import Scenario

from sgd_env.envs import SGDEnv
from sgd_env.envs.generators import default_instance_generator
from sgd_env.policy import ConstantLRPolicy

# dict shared between smac config evaluations
shared_dict = {}

SEED = 42


def evaluate_cost(cfg, seed, instance, **kwargs):
    global shared_dict

    crashed = False
    try:

        instance_idx = int(instance)
        policy_class, env = shared_dict["policy_class"], shared_dict["env"]
        policy = policy_class(**cfg)

        obs = env.reset(instance_idx)
        policy.reset(
            env.instance
        )  # TODO: hack to get to the actual instance / modify interface to support this
        done = False
        total_reward = 0
        print("instance: {}, config: {}".format(env.instance, cfg))
        while not done:
            action = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            crashed = obs["crashed"]
            total_reward += reward

    except Exception:  # catch all exceptions to avoid crashes (make sure the right crash penalty is used)
        traceback.print_exc()
        crashed = True

    cost = -(total_reward if not crashed else env.instance.crash_penalty)
    print("> cost: {}".format(cost))
    return cost


def main():
    global shared_dict

    # 1. Specify training env
    generator = default_instance_generator
    n_training_instances = 1000  # number of instances
    train_env = SGDEnv(
        generator=default_instance_generator, n_instances=n_training_instances
    )

    # 2. Specify policy space
    policy_class = ConstantLRPolicy  # parametric policy family

    cs = ConfigurationSpace()  # config space
    cs.add_hyperparameter(
        UniformFloatHyperparameter("lr", lower=0.000001, upper=10, log=True)
    )

    # Specify AC Scenario
    scenario_dict = {}
    scenario_dict["run_obj"] = "quality"  # we optimize quality
    scenario_dict["runcount-limit"] = 5000  # max # configs evaluated
    scenario_dict["cs"] = cs
    scenario_dict["deterministic"] = True  # env is deterministic
    scenario_dict["instances"] = [[i] for i in range(n_training_instances)]
    scenario_dict["output_dir"] = "results/smac4dac"
    scenario_dict["limit_resources"] = False
    scenario_dict["abort_on_first_run_crash"] = False
    scenario = Scenario(scenario_dict)

    # Add objects required for config evaluation to globals
    shared_dict = {"env": train_env, "policy_class": policy_class}

    # create/run SMAC
    smac = SMAC4AC(
        scenario=scenario,
        rng=SEED,
        tae_runner=evaluate_cost,
        initial_design=RandomConfigurations,
    )

    incumbent = smac.optimize()

    # store incumbent policy
    incumbent_policy = policy_class(**incumbent)
    with open("trained_{}_{}".format(policy_class.__name__, SEED), "w") as fh:
        incumbent_policy.save(fh)


if __name__ == "__main__":
    main()
