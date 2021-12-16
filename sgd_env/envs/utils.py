import functools


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
