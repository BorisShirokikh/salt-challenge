import numpy as np


def fraction(numerator, denominator, empty_val: float = 1):
    assert numerator <= denominator, f'{numerator}, {denominator}'
    return numerator / denominator if denominator != 0 else empty_val


def dice_score(x: np.ndarray, y: np.ndarray) -> float:
    return fraction(2 * np.sum(x & y), np.sum(x) + np.sum(y))

def main_metrics(x: np.ndarray, y: np.ndarray) -> float:
    assert x.shape == y.shape
    if np.sum(x | y) == 0:
        return 1
    elif (np.sum(x) == 0) | (np.sum(y) == 0):
        return 0
    else:
        iou = np.sum(x & y) / np.sum(x | y)
        threshholds = np.arange(0.5, 1, 0.05)
        metric = 0
        for th in threshholds:
            TP = int(th < iou)
            FN = int(np.sum(x & y) < np.sum(y))
            FP = int(np.sum(x & y) < np.sum(x))
            metric += TP / (TP + FN + FP)
        return metric / threshholds.shape[0]
