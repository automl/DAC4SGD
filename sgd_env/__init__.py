from gym.envs.registration import register

register(
    id='sgd-v0',
    entry_point='sgd.envs:SGDEnv',
)
