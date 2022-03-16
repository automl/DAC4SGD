# A Classical Algorithm Configuration Approach to Dynamic Learning Rate Configuration
In this example we statically tune the parameters of hand-crafted dynamic learning rate schedulers using [SMAC](https://github.com/automl/SMAC3).

We illustrate this approach for four different scheduler policies
- ConstantLRPolicy
- CosineAnnealingLRPolicy
- ReduceLROnPlateauPolicy
- SimpleReactivePolicy

(more details see ```description.pdf``` and ```schedulers.py```)

For each of these, we provide tuned configurations (json files in ```tmp/saved_configs```) that can be loaded as follows: 
````batch
policy_cls = ConstantLRPolicy  # (or CosineAnnealingLRPolicy, ReduceLROnPlateauPolicy, SimpleReactivePolicy)
policy = policy_cls.load("tmp/saved_configs")
````

## Reproduction / Retuning

We provide the code we used to configure these schedulers (see ```train.py```). To repeat this process, call:
````
python train.py --policy ConstantLRPolicy --n_instances 1000 --env_seed 42 --smac_seed 42 --smac_budget 5000 --outdir tmp
````
Note:
- The example is for ConstantLRPolicy, for other policies defined in ```schedulers.py``` simply modify the ```--policy``` argument accordingly.
- Running this code require installing [SMAC3](https://github.com/automl/SMAC3) (e.g., calling ```pip install -r requirements_train.txt```).
- This run took 2-3 CPU days on our system, but typically much shorter SMAC runs already returned good configurations.
