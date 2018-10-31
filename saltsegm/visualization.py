import os

import matplotlib.pyplot as plt

from .utils import load_json


# TODO exp_path should contain folders experiment_N, which are to be parsed
def get_points_of_interest(exp_path : str, n_val : int):
    """
    Return max validation metrics and min loss.

    Parameters
    ----------
    exp_path : str
        Path to the experiment folder, containing log.json

    n_val : int
        Validation split number, required to access experiment_{n_val} folder

    Returns
    -------
    tuple (max_val_metrics, min loss) with each element
    being tuple (epoch, value)
    """
    val_path = os.path.join(exp_path, f'experiment_{n_val}')
    # maybe define load_log function?
    log = load_json(os.path.join(val_path, 'log.json'))
    val_losses = log['val_losses']
    val_metrics = log['val_metrics']

    max_metric =(val_metrics.index(max(val_metrics)), max(val_metrics))
    min_loss = (val_losses.index(min(val_losses)), min(val_losses))
    return max_metric, min_loss


# TODO exp_path should contain folders experiment_N, which are to be parsed
def plot_metrics(exp_path : str, n_val : int, highlight=True):
    """
    Plots validation loss, validation metrics and learning rates
        over epochs.

    Parameters
    ----------
    exp_path : str
        Path to the experiment folder, containing log.json

    n_val : int
        Validation split number, required to access experiment_{n_val} folder

    highlight : bool, optional
        Put the max val metrics and min loss points of interest on plot
    """
    val_path = os.path.join(exp_path, f'experiment_{n_val}')
    if highlight:
        max_metric, min_loss = get_points_of_interest(exp_path)

    # maybe define load_log function?
    log = load_json(val_path)
    val_losses = log['val_losses']
    val_metrics = log['val_metrics']
    val_lrs = log['val_lrs']

    fig = plt.figure(figsize=(15,12))
    plt.subplot(3, 1, 1)
    if highlight:
        plt.axvline(min_loss[0], linestyle='--', color='green')
        plt.axhline(min_loss[1], linestyle='--', color='green')
        plt.plot(min_loss[0], min_loss[1], marker='o', markersize=8, color="red")
    plt.plot(val_losses)
    plt.title('Val losses', fontsize=20)

    plt.subplot(3, 1, 2)
    if highlight:
        plt.axvline(max_metric[0], linestyle='--', color='cyan')
        plt.axhline(max_metric[1], linestyle='--', color='cyan')
        plt.plot(max_metric[0], max_metric[1], marker='o', markersize=8, color="red")
    plt.plot(val_metrics)
    plt.title('Val metrics', fontsize=20)

    plt.subplot(3, 1, 3)
    plt.plot(val_lrs)
    plt.title('Learning rates', fontsize=20)
