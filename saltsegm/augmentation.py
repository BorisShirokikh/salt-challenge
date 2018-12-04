import random

import numpy as np


def augm_example(x, y):
    return x, y


def augm_d4(x_, y_):
    """Preforms augmentation using D4 symmetry group"""
    # For notation and naming http://mathstat.slu.edu/escher/index.php/Isometry_Groups
    x_, y_ = np.array(x_), np.array(y_)

    def identity(x, y):
        return x, y

    def rot90(x, y):
        return np.rot90(x, k=1), np.rot90(y, k=1)

    def rot180(x, y):
        return np.rot90(x, k=1), np.rot90(y, k=2)

    def rot270(x, y):
        return np.rot90(x, k=3), np.rot90(y, k=3)

    def m1flip(x, y):
        return np.flip(x, axis=0), np.flip(y, axis=0)

    def m2flip(x, y):
        return np.rot90(np.flip(x, axis=1), axes=(1, 0)), np.rot90(np.flip(y, axis=1), axes=(1, 0))

    def m3flip(x, y):
        return np.flip(x, axis=1), np.flip(y, axis=1)

    def m4flip(x, y):
        return np.flip(np.rot90(x), axis=0), np.flip(np.rot90(y), axis=0)

    augmentations = [identity, rot90, rot180, rot270, m1flip, m2flip, m3flip, m4flip]
    active_augm = random.choice(augmentations)
    return active_augm(x_, y_)


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
