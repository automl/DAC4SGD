from pathlib import Path

from dac4automlcomp.policy import DACPolicy

import schedulers


def load_solution(
    policy_cls=schedulers.ConstantLRPolicy, path=Path("")
) -> DACPolicy:
    """
    Load Solution.

    Serves as an entry point for the competition evaluation.

    By default (the submission) it loads a saved SMAC optimized configuration for the ConstantLRPolicy.

    Args:
        policy_cls: The DACPolicy class object to load
        path: Path pointing to the location the DACPolicy is stored

    Returns
    -------
    DACPolicy
    """
    path /= Path("tmp", "saved_configs")
    return policy_cls.load(path)
