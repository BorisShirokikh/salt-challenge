import random

import numpy as np


def augm_example(x, y):
    return x, y


def augm_mirroring(x, y, prob_to_augm=0.66):
    """Performs mirroring on given `x` and `y`."""
    x, y = np.array(x), np.array(y)
    dims = [-1, -2]

    if np.random.rand() >= prob_to_augm:
        axis_to_reverse = random.choice(dims)

        for i in range(len(x)):
            x[i] = np.flip(x[i], axis_to_reverse)

        for i in range(len(y)):
            y[i] = np.flip(y[i], axis_to_reverse)

    return x, y


def augm_noise(x, noise_ratio=0.10):
    """Performs noise addition to given `x`. Note, that constant feature
    tensors are not transforming."""
    x_np = np.array(x)

    prob_to_augm = 0.5

    if np.random.rand() >= prob_to_augm:
        for i in range(len(x_np)):

            # case of constant features !
            if np.max(x_np[i]) != np.min(x_np[i]):
                x_noise = np.random.uniform(low=1-noise_ratio, high=1+noise_ratio,
                                            size=x_np[i].shape)
                x_np[i] *= x_noise

        # end for

    return x_np
