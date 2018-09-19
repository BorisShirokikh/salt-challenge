import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .utils import load_json, dump_json, load_pred, get_pred
from .torch_models.torch_utils import to_np, to_var, logits2pred
from .dataset import Dataset, DatasetTest


def generate_experiment(exp_path, cv_splits, dataset):
    """Generates experiment with given parameters. Main information saves in config.

    Parameters
    ----------
    exp_path: str
        Path where to generate experiment and save config.

    cv_splits: list
        List of dict(s), which describes cross-val splitting of experiment.

    dataset: class
        Dataset like object.
    """
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    else:
        assert False, f'Experiment `{exp_path}` already exists.'

    config = {'data_path': dataset.data_path,
              'modalities': dataset.modalities,
              'features': dataset.features,
              'target': dataset.target,
              'n_splits': len(cv_splits)}
    dump_json(config, os.path.join(exp_path, 'config.json'))

    for i, split in enumerate(cv_splits):
        val_path = os.path.join(exp_path, f'experiment_{i}')
        os.mkdir(val_path)

        dump_json(list(np.array(split['train_ids'], dtype='str')), os.path.join(val_path, 'train_ids.json'))
        dump_json(list(np.array(split['val_ids'], dtype='str')), os.path.join(val_path, 'val_ids.json'))
        dump_json(list(np.array(split['test_ids'], dtype='str')), os.path.join(val_path, 'test_ids.json'))


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
        dict contaning metrics names and functions.
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
        metric_fn = metrics_dict[metric_name]
        
        results = {}
        for _id, _id_str in zip(test_ids, test_ids_str):
            pred = get_pred(load_pred(_id, pred_path))
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

    del model


def test2predicts(prep_test_path, pred_test_path, model, modalities=['image'], features=None):

    metadata = pd.read_csv(os.path.join(prep_test_path, 'metadata.csv'), index_col=[0])
    test_ids = metadata['id'].values

    ds = DatasetTest(prep_test_path, modalities=modalities, features=features)

    for _id in tqdm(test_ids):
        x = ds.load_x(_id)

        # TODO: add not DL method of prediction

        # *** DL methods ***
        with torch.no_grad():
            model = model.cuda()
            model.eval()

            pred = model(to_var(np.array([x], dtype='float32'), requires_grad=False))
            # TODO: add sequence of models
            pred = to_np(pred)

        # TODO: add postprocessing

        # *** Saving binarized predictions ***
        path_to_save = os.path.join(pred_test_path, f'{_id}.npy')
        np.save(path_to_save, pred[0])


def predicts2csv(pred_test_path, threshold=0.5):
    pass
