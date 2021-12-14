import functools


def random_seeds(rng, n):
    seeds = []
    for _ in range(n):
        seeds.append(rng.randint(1, 4294967295))
    return seeds


def create_instance_func(generator_func, **kwargs):
    return functools.partial(generator_func, **kwargs)


def create_optimizer(optimizer, params, **kwargs):
    return optimizer(params, **kwargs)
