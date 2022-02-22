from gym.envs.registration import register

from sgd_env.envs import generators

register(
    id="sgd-v0",
    entry_point="sgd_env.envs:SGDEnv",
)

__all__ = ["generators"]
