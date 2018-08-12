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