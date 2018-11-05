import os
import json

import numpy as np


def id2png(_id):
    return _id + '.png'


def get_pred(x, threshold=0.5, apply_scaling=False):
    x_spatial = x[0]

    if apply_scaling:
        x_spatial = x_spatial / np.max(x_spatial)

    return x_spatial > threshold


def is_better(cur, best, mode):
    """Checks if `cur` is better than `best` by mode `mode`."""
    assert mode in ('min', 'max'), 'mode should be `max` or `min`'

    if mode == 'min':
        return cur < best
    else:
        return cur > best


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
    
    
def load_config(exp_path: str):
    """Returns config from generated experiment."""
    config_path = os.path.join(exp_path, 'config.json')
    config = load_json(config_path)
    return config


def load_log(exp_path: str, n_val: int):
    """Returns log from n_val split of generated experiment."""
    log_path = os.path.join(exp_path, f'experiment_{n_val}', 'log.json')
    log = load_json(log_path)
    return log


def load_pred(identifier, predictions_path):
    """
    Loads the prediction numpy tensor with specified id.

    Parameters
    ----------
    identifier: int or str
        id to load.

    predictions_path: str
        Path where to load prediction from.

    Returns
    -------
    prediction: numpy.float32
    """
    return np.float32(np.load(os.path.join(predictions_path, f'{identifier}.npy')))


def rl_enc(pred, order='F', return_string=True):
    """Convert binary 2-dim prediction to run-length array or string.

    Parameters
    ----------
    pred: np.ndarray
        2-dim array of predictions

    order: str, optional
        Is down-then-right, i.e. Fortran(F)

    return_string: bool, optional
        Return in `str` or `np.ndarray` dtypes.

    Returns
    -------
    rl_array: str, np.ndarray
        Run-length as a string: <start[1s] length[1s] ... ...> if `return_string` is `True`, else returns
        the same numpy array.
    """
    bytez = pred.reshape(pred.shape[0] * pred.shape[1], order=order)
    bytez = np.concatenate([[0], bytez, [0]])

    runs = np.where(bytez[1:] != bytez[:-1])[0] + 1  # pos start at 1
    runs[1::2] -= runs[::2]

    if return_string:
        rl_array = ' '.join(str(x) for x in runs)
    else:
        rl_array = runs  # not sure about this

    return rl_array
