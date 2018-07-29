import json

import numpy as np


def id2png(_id):
    return _id + '.png'


def get_pred(x, threshold=0.5):
    return x[0] > threshold


def ratio2groups(ratio, splitters=[0.01, 0.25, 0.5, 0.75]):
    """Return classes corresponding to `ratio`, splitted by `splitters`.

    Parameters
    ----------
    ratio: np.ndarray
        Array of target ratios to split into classes.

    splitters: list, optional
        List of values to split the ratio array.

    Returns
    -------
    groups: np.ndarray
        Classes `ratio` was splitted into.
    """
    groups = np.zeros_like(ratio, dtype='int8')

    groups[ratio <= splitters[0]] = 0

    if len(splitters) > 1:
        for i, value in enumerate(splitters[1:]):
            groups[(ratio > splitters[i]) & (ratio <= splitters[i+1])] = i + 1

    groups[ratio > splitters[-1]] = len(splitters)

    return groups


def load_json(path: str):
    """Load the contents of a json file."""
    with open(path, 'r') as f:
        return json.load(f)


def dump_json(value, path: str, *, indent: int = None):
    """Dump a json-serializable object to a json file."""
    with open(path, 'w') as f:
        return json.dump(value, f, indent=indent)
