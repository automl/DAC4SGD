# DAC4SGD

DAC4SGD track of the DAC4AutoML competition at AutoML-Conf.


## Installation

```
git clone git@github.com:automl/dac4automlcomp.git # when using SSH keys to clone
cd dac4automlcomp
pip install -e .
git clone git@github.com/automl/DAC4SGD.git
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

For more see https://github.com/automl/DAC4SGD/tree/main/examples

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

For more policies see https://github.com/automl/DAC4SGD/blob/main/examples/ac_for_dac/schedulers.py

## Sample Submissions
The directory `examples` contains sample submissions with [sample code files](examples/) and an optional [`requirements.txt`](examples/ac_for_dac/requirements.txt) which can contain additional Python packages to be installed *during* an evaluation.

The Bash script [`prepare_upload.sh`](https://github.com/automl/dac4automlcomp/blob/main/prepare_upload.sh) may be used to package a submission directory into a `.zip` file ready for submission to the competition servers.

```
bash prepare_upload.sh <submission_dir>
```

To create a submission, please follow the guidelines [here](https://codalab.lisn.upsaclay.fr/competitions/3672#learn_the_details-evaluation). 

## Evaluating Submissions Locally
[`evaluate_submission.sh`](https://github.com/automl/dac4automlcomp/blob/main/evaluate_submission.sh) can be used to evaluate submissions locally. For example, you can execute the following command to evaluate an example baseline from the DAC4SGD track (where `n` specifies the number of problem instances):

```
bash evaluate_submission.sh -s ../DAC4SGD/examples/ac_for_dac/ -t dac4sgd -n 80
```

NOTE: Please note that if you want to evaluate the experiments in the same runtime environment as the competition servers, you would additionally need to set up the docker container below.

## Docker Container
To run your experiments in the same runtime environment as the competition servers they will be evaluated on, we provide a [Docker](https://docs.docker.com/engine/install/) container. Please see [the Docker container definition file](ubuntu_codalab_Dockerfile.txt) to see what packages will be available in the runtime environment.


The Docker container may be run using the following command:
```
docker run -it -u root raghuspacerajan/dac4automlpy39:latest bash
```

It is not a requirement to use the Docker container to run your local evaluations, but it will be easier for us to debug any issues that may arise when you try evaluating submissions for the competition if you do so inside the provided Docker container.
