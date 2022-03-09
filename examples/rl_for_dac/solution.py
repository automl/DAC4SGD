import os
from dac4automlcomp.policy import DACPolicy
from .policy import SGDPolicy

model_fn = os.path.abspath(os.path.join(os.path.dirname(__file__), "tmp/model.zip"))


def load_solution() -> DACPolicy:
    sgd_policy = SGDPolicy.load(model_fn)
    return sgd_policy
