import functools
from typing import Type

import torch
from gym import spaces


def create_instance_func(generator_func, **kwargs):
    return functools.partial(generator_func, **kwargs)


def create_optimizer(
    optimizer: Type[torch.optim.Optimizer], params, **kwargs
) -> torch.optim.Optimizer:
    return optimizer(params, **kwargs)


def optimizer_action(
    optimizer: torch.optim.Optimizer, name: str, actions: spaces.Dict
) -> None:
    for g in optimizer.param_groups:
        g[name] = actions[name]


def train(model, optimizer, loss_function, loader, steps, device="cpu"):
    model.train()
    for step in range(steps):
        (data, target) = next(loader)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def test(model, loss_function, loader, device="cpu"):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target, reduction="sum").item()
    test_loss /= len(loader.dataset)
    return test_loss
