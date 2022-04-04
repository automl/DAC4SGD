# DAC4SGD

DAC4SGD track of the DAC4AutoML competition at AutoML-Conf.


## Installation

```
# If using SSH keys:
git clone git@github.com:automl-private/dac4automlcomp.git
cd dac4automlcomp
pip install -e .
git clone git@github.com/automl-private/DAC4SGD.git
cd DAC4SGD
pip install -e .
```


## Basic Usage

```python
import gym
import numpy as np
import sgd_env

env = gym.make("sgd-v0", n_instances=np.inf)
env.seed(123)
obs = env.reset()
done = False
lr = 0.001

while not done:
    obs, reward, done, info = env.step(lr)
```

For more see https://github.com/automl-private/DAC4SGD/tree/main/examples

## Using Baseline Policies
```python
import gym
import numpy as np
import sgd_env
from examples.ac_for_dac.schedulers import CosineAnnealingLRPolicy

env = gym.make("sgd-v0", n_instances=np.inf)
env.seed(123)
obs = env.reset()
done = False

lr = 0.01
policy = CosineAnnealingLRPolicy(lr)
policy.reset(env.current_instance)

while not done:
    lr = policy.act(obs)
    obs, reward, done, info = env.step(lr)
```

For more policies see https://github.com/automl-private/DAC4SGD/blob/main/examples/ac_for_dac/schedulers.py
