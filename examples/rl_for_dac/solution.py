import os
from pathlib import Path
from typing import Union

from dac4automlcomp.policy import DACPolicy

from policy import SGDPolicy


def load_solution(path: Union[str, Path] = Path(".")) -> DACPolicy:
    path = Path(path)
    model_fn = path / "tmp/model.zip"
    sgd_policy = SGDPolicy.load(model_fn)
    return sgd_policy
