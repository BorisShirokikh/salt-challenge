import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from skimage.transform import resize

from .utils import load_json, dump_json, load_pred, get_pred, rl_enc, load
from .torch_models.torch_utils import to_np, to_var, logits2pred
from .torch_models.model import fit_model
from .dataset import Dataset, DatasetTest


def generate_experiment(exp_path, cv_splits, dataset, exp_type='tvt'):
    """Generates experiment with given parameters. Main information saves in config.

    Parameters
    ----------
    exp_path: str
        Path where to generate experiment and save config.

    cv_splits: list
        List of dict(s), which describes cross-val splitting of experiment.

    dataset: class
        Dataset like object.

    exp_type: str
        Type of experiment: train-val-test (`tvt`) or train-val (`tv`).
    """
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    else:
        assert False, f'Experiment `{exp_path}` already exists.'

    assert exp_type in ('tvt', 'tv'), \
        f'experiment type should be `tvt` or `tv`, {exp_type} given'

    config = {'data_path': dataset.data_path,
              'modalities': dataset.modalities,
              'features': dataset.features,
              'target': dataset.target,
              'n_splits': len(cv_splits),
              'exp_type': exp_type}
    dump_json(config, os.path.join(exp_path, 'config.json'))

    for i, split in enumerate(cv_splits):
        val_path = os.path.join(exp_path, f'experiment_{i}')
        os.mkdir(val_path)

        dump_json(list(np.array(split['train_ids'], dtype='str')), os.path.join(val_path, 'train_ids.json'))
        dump_json(list(np.array(split['val_ids'], dtype='str')), os.path.join(val_path, 'val_ids.json'))

        if exp_type == 'tvt':
            dump_json(list(np.array(split['test_ids'], dtype='str')), os.path.join(val_path, 'test_ids.json'))
        elif exp_type == 'tv':
            pass


def load_val_data(exp_path, n_val):
    """Loads stacks of images to validate model on.

    Parameters
    ----------
    exp_path: str
        Path where to load experiment info from.

    n_val: int
        The id of validation (depends on number of generated experiments).

    Returns
    -------
        x_val, y_val: np.ndarray
    """
    config_path = os.path.join(exp_path, 'config.json')

    config = load_json(config_path)
    ds = Dataset(data_path=config['data_path'], modalities=config['modalities'],
                 features=config['features'], target=config['target'])

    val_path = os.path.join(exp_path, f'experiment_{n_val}')

    val_ids = np.array(load_json(os.path.join(val_path, 'val_ids.json')), dtype='int64')

    x_val, y_val = [], []
    for _id in val_ids:
        x_val.append(ds.load_x(_id))
        y_val.append(ds.load_y(_id))
    x_val = np.array(x_val, dtype='float32')
    y_val = np.array(y_val, dtype='float32')

    return x_val, y_val


def calculate_metrics(exp_path, n_val, metrics_dict):
    """Calculates and saves test metric values in `test_metrics` folder.

    Parameters
    ----------
    exp_path: str
        Path to the experiment.

    n_val: int
        The id of cross-val to calculates metrics in.

    metrics_dict: dict
        dict containing metrics names which map into (`function`, `apply_scaling`).
        `apply_scaling` is `bool` value indicates apply or not the scaling on prediction.
    """
    config_path = os.path.join(exp_path, 'config.json')

    config = load_json(config_path)
    ds = Dataset(data_path=config['data_path'], modalities=config['modalities'],
                 features=config['features'], target=config['target'])

    val_path = os.path.join(exp_path, f'experiment_{n_val}')
    pred_path = os.path.join(val_path, 'test_predictions')

    metric_path = os.path.join(val_path, 'test_metrics')
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)

    test_ids_str = load_json(os.path.join(val_path, 'test_ids.json'))
    test_ids = np.array(test_ids_str, dtype='int64')

    for metric_name in metrics_dict.keys():
        metric_fn, apply_scaling = metrics_dict[metric_name]

        results = {}
        for _id, _id_str in zip(test_ids, test_ids_str):
            pred = get_pred(load_pred(_id, pred_path), apply_scaling=apply_scaling)
            mask = get_pred(ds.load_y(_id))

            result = metric_fn(mask, pred)
            results[_id_str] = result
        # end for

        metric_filename = os.path.join(metric_path, metric_name + '.json')
        assert not os.path.exists(metric_filename), \
            f'metric {metric_name} has already been calculated'
        dump_json(results, metric_filename)
    # end for


def get_experiment_result(exp_path, n_splits, metric_name):
    val_results = []

    for i in range(n_splits):
        metric_path = os.path.join(exp_path, f'experiment_{i}/test_metrics/{metric_name}.json')
        results_dict = load_json(metric_path)

        val_mean = np.mean(list(results_dict.values()))
        val_results.append(val_mean)
    # end for

    return np.mean(val_results)


def make_predictions(exp_path, n_val):
    """Makes test predictions and saves them in `test_predictions` folder.

    Parameters
    ----------
    exp_path: str
        Path to the experiment.

    n_val: int
        The id of cross-val to make predictions in.
    """
    config_path = os.path.join(exp_path, 'config.json')

    config = load_json(config_path)
    ds = Dataset(data_path=config['data_path'], modalities=config['modalities'],
                 features=config['features'], target=config['target'])

    val_path = os.path.join(exp_path, f'experiment_{n_val}')

    model = torch.load(os.path.join(val_path, 'model.pt'))
    model = model.cuda()

    pred_path = os.path.join(val_path, 'test_predictions')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    test_ids_str = load_json(os.path.join(val_path, 'test_ids.json'))
    test_ids = np.array(test_ids_str, dtype='int64')

    for _id, _id_str in zip(test_ids, test_ids_str):
        x = ds.load_x(_id)

        # DO INFERENCE STEP:
        with torch.no_grad():
            model.eval()
            x_t = to_var(np.array([x], dtype='float32'), requires_grad=False)
            y = to_np(logits2pred(model(x_t)))[0]

            y_filename = os.path.join(pred_path, _id_str + '.npy')

            np.save(y_filename, y)

            del x, x_t, y
    # end for


def do_experiment(exp_path, n_val):
    """Performs learning, making predictions and calculating metrics processes."""

    print('>>> loading resources..')

    val_path = os.path.join(exp_path, f'experiment_{n_val}')
    resources = load(os.path.join(val_path, 'resources.gz'))

    torch_model = resources['torch_model']
    batch_iter = resources['batch_iter']
    epochs = resources['epochs']
    steps_per_epoch = resources['steps_per_epoch']
    saving_model_mode = resources['saving_model_mode']
    metrics_dict = resources['metrics_dict']

    val_data = load_val_data(exp_path=exp_path, n_val=n_val)

    print('>>> fitting model..')

    fit_model(
        torch_model=torch_model, generator=batch_iter.flow(), val_path=val_path, val_data=val_data,
        epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=True,
        saving_model_mode=saving_model_mode
    )

    print('>>> making predictions..')
    make_predictions(exp_path=exp_path, n_val=n_val)

    print('>>> calculating metrics..')
    calculate_metrics(exp_path=exp_path, n_val=n_val, metrics_dict=metrics_dict)


def test2csv_pred(prep_test_path, csv_filename, model, modalities=['image'], features=None,
                  threshold=0.5):
    """Converts test images with given `model` into submission-ready csv-file.

    Parameters
    ----------
    prep_test_path: str
        Path to stored test data with generated `metadata`.

    csv_filename: str
        Filename of csv-file to save. Should have 'some_name.csv' structure.

    model: torch.nn.Module, or the same
        Model to do predictions with.

    modalities: list, optional
        List of modalities to load as channel(s) of the single image.

    features: list, optional
        List of features to load as additional channel(s) of the single image.

    threshold: float
        Threshold to make binary prediction. Must be between 0. and 1.
    """
    metadata = pd.read_csv(os.path.join(prep_test_path, 'metadata.csv'), index_col=[0])
    test_ids = metadata.index
    test_ids_orig = metadata['id'].values

    ds = DatasetTest(prep_test_path, modalities=modalities, features=features)

    id_rle_dict = {}

    for _id in tqdm(test_ids):
        x = ds.load_x(_id)

        # TODO: add not DL method of prediction

        # *** DL methods ***
        with torch.no_grad():
            model = model.cuda()
            model.eval()

            pred = model(to_var(np.array([x], dtype='float32'), requires_grad=False))
            # TODO: add sequence of models
            pred = to_np(pred)[0]

        # TODO: add postprocessing

        # converting to 2-dim numpy array (image-like) of original shape (101x101)
        # then to binary predictions
        if len(pred.shape) == 3:
            pred = pred[0]

        ORIG_SIZE = 101
        if pred.shape[-1] != ORIG_SIZE:
            pred = resize(pred, output_shape=(ORIG_SIZE, ORIG_SIZE), order=3, preserve_range=True)

        pred = pred > threshold

        # *** Encoding binarized predictions ***
        rle_pred = rl_enc(pred)
        id_rle_dict[test_ids_orig[_id]] = rle_pred

    # end for

    # *** Saving submission csv-file ***
    csv_to_save = pd.DataFrame.from_dict(id_rle_dict, orient='index')

    csv_to_save.index.names = ['id']
    csv_to_save.columns = ['rle_mask']

    csv_to_save.to_csv(os.path.join(prep_test_path, csv_filename))
