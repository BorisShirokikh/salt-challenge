import numpy as np


def fraction(numerator, denominator, empty_val: float = 1):
    assert numerator <= denominator, f'{numerator}, {denominator}'
    return numerator / denominator if denominator != 0 else empty_val


def dice_score(x: np.ndarray, y: np.ndarray) -> float:
    return fraction(2 * np.sum(x & y), np.sum(x) + np.sum(y))
