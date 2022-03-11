import os
import sys
from functools import wraps

import torch
from gym import spaces


def optimizer_action(
    optimizer: torch.optim.Optimizer, name: str, actions: spaces.Dict
) -> None:
    for g in optimizer.param_groups:
        g[name] = actions[name]


def train(model, optimizer, loss_function, loader, steps, device="cpu"):
    """Optimize given `model` for `loss_function` using `optimizer` for `steps` steps.

    Returns:
        loss: Final mini batch training loss per data point
    """
    model.train()
    for step in range(steps):
        (data, target) = next(loader)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target, reduction="none")
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
    return loss


def test(model, loss_function, loader, device="cpu"):
    """Evaluate given `model` on `loss_function`.

    Returns:
        test_losses: Full batch validation loss per data point
    """
    model.eval()
    test_losses = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_losses.append(loss_function(output, target, reduction="none"))
    test_losses = torch.cat(test_losses)
    return test_losses


def supress_output(func):
    """Wrapper to supress stdout of the `func`"""

    @wraps(func)
    def wrapped(*args, **kwargs):
        f = open(os.devnull, "w")
        sys.stdout = f
        out = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return out

    return wrapped
