import os

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, data_path, modalities=['image'], features=None, target='target'):
        """Class of dataset, that supports the core methods to load ids, images and metadata.

        Parameters
        ----------
        data_path: str
            Path to preprocess data with generated metadata.

        modalities: list
            Names of modalities in metadata to load into x.

        features: list, None, optional
            Names of features in metadata to load into x.

        target: str
            Name of target in metadata to load into y.
        """
        self.data_path = data_path
        self.modalities = modalities
        self.features = features
        self.target = target

        metadata_filename = os.path.join(data_path, 'metadata.csv')
        assert os.path.exists(metadata_filename), f'There is no such file {metadata_filename}'

        self.metadata = pd.read_csv(metadata_filename, index_col=[0])

        self.ids = self.metadata.index
        self.names = self.metadata['id'].values

        n_modalities = len(modalities)

        self.features_max_values = []

        self.n_channels = n_modalities

        if features is None:
            self.features_max_values = None
        else:
            n_features = len(features)
            self.n_channels += n_features

    def load_x(self, _id):
        """Returns ndarray of x corresponding to given `_id`."""
        assert _id in self.ids, f'There is no such id ({_id}) in dataset'

        modal_paths = []
        for modal in self.modalities:
            modal_path = os.path.join( self.data_path, self.metadata.iloc[_id][modal] )
            modal_paths.append(modal_path)

        modality_shape = np.load(modal_paths[0]).shape

        xs = []
        for modal_path in modal_paths:
            x = np.load(modal_path)
            assert x.shape == modality_shape, 'All modalities should have equal shapes'

            xs.append(x)

        if self.features is not None:
            for feature in self.features:
                feature_value = self.metadata.iloc[_id][feature]
                x = np.zeros(modality_shape) + feature_value
                xs.append(x)

        return np.array(xs, dtype='float32')

    def load_y(self, _id):
        """Returns ndarray of y corresponding to given `_id`."""
        assert _id in self.ids, f'There is no such id ({_id}) in dataset'

        target_content = self.metadata.iloc[_id][self.target]

        if isinstance(target_content, str):  # means `path`
            filename = os.path.join(self.data_path, target_content)
            y = np.load(filename)
        else:  # means some feature, e.g. `target_ratio`
            y = [[target_content]]

        return np.array([y], dtype='float32')


class DatasetTest:
    def __init__(self, data_path, modalities=['image'], features=None):
        """Class of dataset for test data, that supports the method to load modalities images and features.

        Parameters
        ----------
        data_path: str
            Path to preprocess data with generated metadata.

        modalities: list
            Names of modalities in metadata to load into x.

        features: list, None, optional
            Names of features in metadata to load into x.
        """
        self.data_path = data_path
        self.modalities = modalities
        self.features = features

        metadata_filename = os.path.join(data_path, 'metadata.csv')
        assert os.path.exists(metadata_filename), f'There is no such file {metadata_filename}'

        self.metadata = pd.read_csv(metadata_filename, index_col=[0])

        self.ids = self.metadata.index
        self.names = self.metadata['id'].values

        n_modalities = len(modalities)

        self.features_max_values = []

        self.n_channels = n_modalities

        if features is None:
            self.features_max_values = None
        else:
            n_features = len(features)
            self.n_channels += n_features

    def load_x(self, _id):
        """Returns ndarray of x corresponding to given `_id`."""
        assert _id in self.ids, f'There is no such id ({_id}) in dataset'

        modal_paths = []
        for modal in self.modalities:
            modal_path = os.path.join( self.data_path, self.metadata.iloc[_id][modal] )
            modal_paths.append(modal_path)

        modality_shape = np.load(modal_paths[0]).shape

        xs = []
        for modal_path in modal_paths:
            x = np.load(modal_path)
            assert x.shape == modality_shape, 'All modalities should have equal shapes'

            xs.append(x)

        if self.features is not None:
            for feature in self.features:
                feature_value = self.metadata.iloc[_id][feature]
                x = np.zeros(modality_shape) + feature_value
                xs.append(x)

        return np.array(xs, dtype='float32')


def filter_notarget_ids(ids, df):
    filtered_ids = ids[df.target_ratio > 0]
    return filtered_ids
