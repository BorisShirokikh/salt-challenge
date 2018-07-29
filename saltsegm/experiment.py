import os

import numpy as np

from .utils import load_json, dump_json

def generate_experiment(exp_path, cv_splits, dataset):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    dataset_info = {'data_path': dataset.data_path,
                    'modalities': dataset.modalities,
                    'target': dataset.target}
    dump_json( dataset_info, os.path.join(exp_path, 'dataset.json') )

    n_val = len(cv_splits)
    for i, split in enumerate(cv_splits):
        val_path = os.path.join(exp_path, f'experiment_{i}')
        os.mkdir(val_path)

        dump_json( list(np.array(split['train_ids'], dtype='str')), os.path.join(val_path, 'train_ids.json') )
        dump_json( list(np.array(split['val_ids'], dtype='str')), os.path.join(val_path, 'val_ids.json') )
        dump_json( list(np.array(split['test_ids'], dtype='str')), os.path.join(val_path, 'test_ids.json') )
