import matplotlib.pyplot as plt

from .utils import load_log


# TODO exp_path should contain folders experiment_N, which are to be parsed
def get_points_of_interest(exp_path: str, n_val: int):
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

    log = load_log(exp_path, n_val)
    val_losses = log['val_losses']
    val_metrics = log['val_metrics']

    max_metric = (val_metrics.index(max(val_metrics)), max(val_metrics))
    min_loss = (val_losses.index(min(val_losses)), min(val_losses))
    return max_metric, min_loss


# TODO exp_path should contain folders experiment_N, which are to be parsed
def plot_metrics(exp_path: str, n_val: int, fig_size=(15, 12), highlight=True):
    """
    Plots validation loss, validation metrics and learning rates
        of given n_val split over epochs.

    Parameters
    ----------
    exp_path : str
        Path to the experiment folder, containing log.json

    n_val : int
        Validation split number, required to access experiment_{n_val} folder

    fig_size : (int, int), optional
        Size of canvas for plots.

    highlight : bool, optional
        Put the max val metrics and min loss points of interest on plot
    """
    fig, ax = plt.subplots(3, 1, figsize=fig_size)

    log = load_log(exp_path, n_val)
    val_losses = log['val_losses']
    val_metrics = log['val_metrics']
    val_lrs = log['val_lrs']

    # CAREFUL WITH FONTSIZE! Labels may overlap!
    ax[0].set_title('Val losses', fontsize=20)
    ax[0].plot(val_losses)

    ax[1].set_title('Val metrics', fontsize=20)
    ax[1].plot(val_metrics)

    ax[2].set_title('Learning rates', fontsize=20)
    ax[2].plot(val_lrs)

    # TODO: enhance text positioning
    if highlight:
        max_metric, min_loss = get_points_of_interest(exp_path, n_val)

        ax[0].axvline(min_loss[0], linestyle='--', color='cyan')
        ax[0].axhline(min_loss[1], linestyle='--', color='cyan')
        ax[0].plot(min_loss[0], min_loss[1], marker='o', markersize=8, color="red")
        ax[0].text(min_loss[0], min_loss[1]+0.1, f'{min_loss[0]}, {min_loss[1]: 0.2f}', fontsize=12)

        ax[1].axvline(max_metric[0], linestyle='--', color='cyan')
        ax[1].axhline(max_metric[1], linestyle='--', color='cyan')
        ax[1].plot(max_metric[0], max_metric[1], marker='o', markersize=8, color="red")
        ax[1].text(max_metric[0], max_metric[1] - 0.15, f'{max_metric[0]}, {max_metric[1]: 0.2f}', fontsize=12)

    plt.plot()