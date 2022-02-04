import numpy as np
import torch
from gym import spaces


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
        loss = loss_function(output, target, reduction="none")
        optimizer.zero_grad()
        loss.mean().backward()
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


def generate_seed(idx, rng):
    if rng is not generate_seed.rng:
        generate_seed.cache = []
        generate_seed.rng = rng
    while idx >= len(generate_seed.cache):
        seed = rng.randint(1, 4294967295, dtype=np.int64)
        generate_seed.cache.append(seed)
    return generate_seed.cache[idx]


generate_seed.rng = None


def get_instance(generator, idx, rng):
    seed = generate_seed(idx, rng)
    rng = np.random.RandomState(seed)
    instance = generator(rng)
    return instance, seed
