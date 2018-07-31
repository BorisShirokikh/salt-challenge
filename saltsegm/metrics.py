import numpy as np


def fraction(numerator, denominator, empty_val: float = 1):
    assert numerator <= denominator, f'{numerator}, {denominator}'
    return numerator / denominator if denominator != 0 else empty_val


def dice_score(x: np.ndarray, y: np.ndarray) -> float:
    assert x.dtype == bool and y.dtype == bool, \
        'input array should have bool dtype, {x.dtype} {y.dtype} are given'
        
    assert x.shape == y.shape
        
    return fraction(2 * np.sum(x & y), np.sum(x) + np.sum(y))


def main_metric(true: np.ndarray, pred: np.ndarray) -> float:
    assert true.dtype == bool and pred.dtype == bool, \
        'input array should have bool dtype, {true.dtype} {pred.dtype} are given'
        
    assert true.shape == pred.shape
    
    if np.sum(true | pred) == 0:
        return 1
    elif (np.sum(true) == 0) or (np.sum(pred) == 0):
        return 0
    else:
        iou = np.sum(true & pred) / np.sum(true | pred)
        threshholds = np.arange(0.5, 1, 0.05)
        metric = 0
        for th in threshholds:
            TP = int(th < iou)
            FN = int(th > iou)
            FP = int(th > iou)
            metric += TP / (TP + FN + FP)
        return metric / threshholds.shape[0]
