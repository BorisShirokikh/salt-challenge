import os

import numpy as np

from .utils import load_json, dump_json
from .dataset import Dataset

def generate_experiment(exp_path, cv_splits, dataset):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    config = {'data_path': dataset.data_path,
              'modalities': dataset.modalities,
              'target': dataset.target,
              'n_splits': len(cv_splits)}
    dump_json( config, os.path.join(exp_path, 'config.json') )

    for i, split in enumerate(cv_splits):
        val_path = os.path.join(exp_path, f'experiment_{i}')
        os.mkdir(val_path)

        dump_json( list(np.array(split['train_ids'], dtype='str')), os.path.join(val_path, 'train_ids.json') )
        dump_json( list(np.array(split['val_ids'], dtype='str')), os.path.join(val_path, 'val_ids.json') )
        dump_json( list(np.array(split['test_ids'], dtype='str')), os.path.join(val_path, 'test_ids.json') )


def load_experiment(exp_path, n_val):
    config_path = os.path.join(exp_path, 'config.json')

    config = load_json(config_path)
    ds = Dataset(data_path=config['data_path'], modalities=config['modalities'],
                 target=config['target'])

    val_path = os.path.join(exp_path, f'experiment_{n_val}')

    train_ids = np.array( load_json(os.path.join(val_path, 'train_ids.json')), dtype='int64' )
    val_ids = np.array( load_json(os.path.join(val_path, 'val_ids.json')), dtype='int64' )
    test_ids = np.array( load_json(os.path.join(val_path, 'test_ids.json')), dtype='int64' )

    x_train, y_train = [], []
    for _id in train_ids:
        x_train.append( ds.load_x(_id) )
        y_train.append( ds.load_y(_id) )
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_val, y_val = [], []
    for _id in val_ids:
        x_val.append( ds.load_x(_id) )
        y_val.append( ds.load_y(_id) )
    x_val, y_val = np.array(x_val), np.array(y_val)

    x_test, y_test = [], []
    for _id in train_ids:
        x_test.append( ds.load_x(_id) )
        y_test.append( ds.load_y(_id) )
    x_test, y_test = np.array(x_test), np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test        
            