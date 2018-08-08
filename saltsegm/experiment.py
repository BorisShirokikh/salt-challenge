import os

import numpy as np
from tqdm import tqdm

from .utils import load_json, dump_json, load_pred, get_pred
from .dataset import Dataset


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

    config = {'data_path': dataset.data_path,
              'modalities': dataset.modalities,
              'target': dataset.target,
              'n_splits': len(cv_splits)}
    dump_json(config, os.path.join(exp_path, 'config.json'))

    for i, split in enumerate(cv_splits):
        val_path = os.path.join(exp_path, f'experiment_{i}')
        os.mkdir(val_path)

        dump_json(list(np.array(split['train_ids'], dtype='str')), os.path.join(val_path, 'train_ids.json'))
        dump_json(list(np.array(split['val_ids'], dtype='str')), os.path.join(val_path, 'val_ids.json'))
        dump_json(list(np.array(split['test_ids'], dtype='str')), os.path.join(val_path, 'test_ids.json'))


def load_experiment_data(exp_path, n_val):
    """Loads stacks of images to carry on experiment with.

    Parameters
    ----------
    exp_path: str
        Path where to load experiment info from.

    n_val: int
        The id of validation (depends on number of generated experiments).

    Returns
    -------
        x_train, y_train, x_val, y_val, x_test, y_test: np.ndarray
    """
    config_path = os.path.join(exp_path, 'config.json')

    config = load_json(config_path)
    ds = Dataset(data_path=config['data_path'], modalities=config['modalities'],
                 target=config['target'])

    val_path = os.path.join(exp_path, f'experiment_{n_val}')

    train_ids = np.array(load_json(os.path.join(val_path, 'train_ids.json')), dtype='int64')
    val_ids = np.array(load_json(os.path.join(val_path, 'val_ids.json')), dtype='int64')
    test_ids = np.array(load_json(os.path.join(val_path, 'test_ids.json')), dtype='int64')

    x_train, y_train = [], []
    for _id in train_ids:
        x_train.append(ds.load_x(_id))
        y_train.append(ds.load_y(_id))
    x_train = np.array(x_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')

    x_val, y_val = [], []
    for _id in val_ids:
        x_val.append(ds.load_x(_id))
        y_val.append(ds.load_y(_id))
    x_val = np.array(x_val, dtype='float32')
    y_val = np.array(y_val, dtype='float32')

    x_test, y_test = [], []
    for _id in test_ids:
        x_test.append(ds.load_x(_id))
        y_test.append(ds.load_y(_id))
    x_test = np.array(x_test, dtype='float32')
    y_test = np.array(y_test, dtype='float32')

    return x_train, y_train, x_val, y_val, x_test, y_test


def make_predictions(exp_path, n_val, model):
    config_path = os.path.join(exp_path, 'config.json')

    config = load_json(config_path)
    ds = Dataset(data_path=config['data_path'], modalities=config['modalities'],
                 target=config['target'])

    val_path = os.path.join(exp_path, f'experiment_{n_val}')

    pred_path = os.path.join(val_path, 'test_predictions')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    test_ids_str = load_json(os.path.join(val_path, 'test_ids.json'))
    test_ids = np.array(test_ids_str, dtype='int64')

    for _id, _id_str in tqdm(zip(test_ids, test_ids_str)):
        x = ds.load_x(_id)
        y = model.do_inf_step([x])[0]

        y_filename = os.path.join(pred_path, _id_str + '.npy')

        np.save(y_filename, y)


def calculate_metrics(exp_path, n_val, metrics_dict):
    config_path = os.path.join(exp_path, 'config.json')

    config = load_json(config_path)
    ds = Dataset(data_path=config['data_path'], modalities=config['modalities'],
                 target=config['target'])
    
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
        dump_json(results, metric_filename)
    # end for
