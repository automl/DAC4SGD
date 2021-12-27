import functools

import torch


def create_instance_func(generator_func, **kwargs):
    return functools.partial(generator_func, **kwargs)


def create_optimizer(optimizer, params, **kwargs):
    return optimizer(params, **kwargs)


def train(model, optimizer, loss_function, loader, steps):
    model.train()
    for step in range(steps):
        (data, target) = loader.next()
        output = model(data)
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def test(model, loss_function, loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            test_loss += loss_function(output, target, reduction="sum").item()
    test_loss /= len(loader.dataset)
    return test_loss
