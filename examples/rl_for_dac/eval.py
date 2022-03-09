import numpy as np
import gym
from examples.utils import run_policy
from policy import SGDPolicy


if __name__ == '__main__':
    # Load model and env
    model_fn = "tmp/model.zip"
    sgd_policy = SGDPolicy.load(model_fn)

    # Test with original env
    env = gym.make("sgd-v0", n_instances=sgd_policy.args.n_instances)
    env.seed(sgd_policy.args.seed)

    # Exemplarily evaluate
    states, rewards = run_policy(env=env, policy=sgd_policy, instance=None)
    cumulative_reward = np.sum(rewards)
    print("Cumulative Reward: ", cumulative_reward)
