# Learn Dynamic Hyperparameters with RL
In this example the learning rate of the `sgd_env` is dynamically adapted with a 
learned Reinforcement Learning agent.

To train the agent, call
```bash
python train.py --n_steps 1000
```
with 1000 steps (which should be a larger number in practice).

In `eval.py` a trained agent is loaded, converted into the required policy format
and evaluated.

## Reproduce Model
In order to reproduce the model run 
```bash
python train.py --n_steps 1000000 --seed 54321 --agent A2C
```