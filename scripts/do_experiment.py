import os
import argparse
import warnings

from saltsegm.experiment import do_experiment

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', required=True, type=str)
    parser.add_argument('--n_val', required=True, type=int)
    args = parser.parse_known_args()[0]

    exp_path = args.exp_path
    n_val = args.n_val
    val_path = os.path.join(exp_path, f'experiment_{n_val}')
    print('> experiment:', val_path)

    do_experiment(exp_path=exp_path, n_val=n_val)
