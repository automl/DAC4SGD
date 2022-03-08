# DAC4SGD

## Requirements

```
numpy
torch
torchvision
gym
configspace
```

## Installation

```bash
git clone https://github.com/automl-private/dac4automlcomp/
cd dac4automlcomp
python setup.py install --user
cd ..
git clone https://github.com/automl-private/DAC4SGD
cd DAC4SGD
python setup.py install --user
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
from sgd_env.policy.schedulers import CosineAnnealingLRPolicy

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

For more policies see https://github.com/automl-private/DAC4SGD/blob/main/sgd_env/policy
