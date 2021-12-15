import functools


def create_instance_func(generator_func, **kwargs):
    return functools.partial(generator_func, **kwargs)


def create_optimizer(optimizer, params, **kwargs):
    return optimizer(params, **kwargs)
