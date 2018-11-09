import random

import numpy as np
from tqdm import tqdm

from .augmentation import augm_mirroring, augm_noise


class BatchIter:
    def __init__(self, train_ids, load_x, load_y, batch_size=32,
                 mirroring_augm_prob=0.66, noise_augm_ratio=0.1,
                 verbose_loading=True):
        """Creates batch iterator both for keras and torch models.

        Parameters
        ----------
        train_ids: list
            ids to generate train batches.

        load_x: Callable
            `dataset` function to load images.

        load_y: Callable
            `dataset` function to load masks.

        batch_size: int, optional
            Size of generated batch.

        mirroring_augm_prob: float, optional
            Probability to apply mirroring augmentation. Uses function
            `saltsegm.augmentation.augm_mirroring`.
            
        noise_augm_ratio: float, optional
            Ratio of noise applying to `x`. Uses function
            `saltsegm.augmentation.augm_noise`.

        verbose_loading: bool, optional
            If `True`, showing progress of data loading.
        """
        self.train_ids = train_ids

        self.x_size = load_x(train_ids[0]).shape
        self.y_size = load_y(train_ids[0]).shape

        self.batch_size = batch_size

        x_stack = []
        y_stack = []

        if verbose_loading:
            for _id in tqdm(train_ids):
                x_stack.append(load_x(_id))
                y_stack.append(load_y(_id))
        else:
            for _id in train_ids:
                x_stack.append(load_x(_id))
                y_stack.append(load_y(_id))

        self.x_stack = np.array(x_stack, dtype='float32')
        self.y_stack = np.array(y_stack, dtype='float32')

        assert mirroring_augm_prob >= 0 and mirroring_augm_prob <= 1, \
            f'probability should be between 0 and 1, {mirroring_augm_prob} given'
        self.mirroring_augm_prob = mirroring_augm_prob

        assert noise_augm_ratio >= 0 and noise_augm_ratio <= 1, \
            f'Noise ratio should be between 0 and 1, {noise_augm_ratio} given'
        self.noise_augm_ratio = noise_augm_ratio

        self.inner_ids = np.arange(len(train_ids))

    def flow(self):
        """Generator."""
        batch_x = np.array([np.zeros(self.x_size)] * self.batch_size, dtype='float32')
        batch_y = np.array([np.zeros(self.y_size)] * self.batch_size, dtype='float32')

        while True:
            for i in range(self.batch_size):
                index = random.choice(self.inner_ids)
                x = self.x_stack[index]
                y = self.y_stack[index]

                batch_x[i] = x
                batch_y[i] = y

#                 x_mirrored, batch_y[i] = augm_mirroring(
#                     x, y, prob_to_augm=self.mirroring_augm_prob
#                 )

#                 batch_x[i] = augm_noise(
#                     x_mirrored, noise_ratio=self.noise_augm_ratio
#                 )

            yield batch_x, batch_y
