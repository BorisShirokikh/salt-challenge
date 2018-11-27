import os

from ipywidgets import IntSlider, interact
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from .utils import load_log, load_json, get_pred
from .dataset import Dataset


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
        Validation split number, required to access experiment_{n_val} folder.

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
        ax[0].text(0.75, 0.75, f'{min_loss[0]}, {min_loss[1]: 0.3f}',
                   horizontalalignment='center', verticalalignment='center',
                   transform = ax[0].transAxes, fontsize=14)
        # ax[0].figtext(0, 1, f'{min_loss[0]}, {min_loss[1]: 0.3f}', fontsize=12)
        
        ax[1].axvline(max_metric[0], linestyle='--', color='cyan')
        ax[1].axhline(max_metric[1], linestyle='--', color='cyan')
        ax[1].plot(max_metric[0], max_metric[1], marker='o', markersize=8, color="red")
        # ax[1].text(max_metric[0], max_metric[1] - 0.1, f'{max_metric[0]}, {max_metric[1]: 0.2f}', fontsize=12)
        ax[1].text(0.75, 0.25, f'{max_metric[0]}, {max_metric[1]: 0.3f}',
                   horizontalalignment='center', verticalalignment='center',
                   transform = ax[1].transAxes, fontsize=14)

    plt.plot()
    plt.show()


# TODO sync threshold here and in metrics
def show_predictions(data_path: str, exp_path: str, n_val: int, threshold=0.5):
    """
    Show results in form of
        Probability map / Prediction / Scaled Prediction / Mask

    Parameters
    ----------
    data_path : str
        Path to train dataset

    exp_path : str
        Path to experiment folder

    n_val : int
        Validation split number

    threshold : int, optional
        Threshold value for predictions
    """
    ds_train = Dataset(data_path)

    test_ids_path = os.path.join(exp_path, f'experiment_{n_val}', 'test_ids.json')
    test_ids = load_json(test_ids_path)
    masks_folder = os.path.join(exp_path, f'experiment_{n_val}', 'test_predictions')

    prob_maps, preds, sc_preds, masks = [], [], [], []

    for _id in test_ids:
        img_path = os.path.join(masks_folder, str(_id) + '.npy')
        prob_maps.append(np.load(img_path))
        preds.append(get_pred(prob_maps[-1], threshold=threshold, apply_scaling=False))
        sc_preds.append(get_pred(prob_maps[-1], threshold=threshold, apply_scaling=True))
        masks.append(ds_train.load_y(int(_id)))
    # end for

    def id2image(idx):
        fig, axs = plt.subplots(1, 4, figsize=(10, 5))

        ax_prob = axs[0]
        ax_pred = axs[1]
        ax_pred_sc = axs[2]
        ax_mask = axs[3]

        ax_prob.set_title('prob_map', fontsize=12)
        ax_prob.imshow(prob_maps[idx].squeeze())

        ax_pred.set_title('prediction', fontsize=12)
        ax_pred.imshow(preds[idx])

        ax_pred_sc.set_title('scaled prediction', fontsize=12)
        ax_pred_sc.imshow(sc_preds[idx])

        ax_mask.set_title('target_mask', fontsize=12)
        ax_mask.imshow(masks[idx].squeeze())

        plt.show()

        print(f'id: {test_ids[idx]}')

    sld = IntSlider(min=0, max=len(test_ids) - 1, step=1, continuous_update=False)
    interact(id2image, idx=sld)
