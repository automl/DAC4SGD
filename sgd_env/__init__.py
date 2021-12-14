from gym.envs.registration import register

from .envs import hyperparameters

register(
    id="sgd-v0",
    entry_point="sgd_env.envs:SGDEnv",
)
