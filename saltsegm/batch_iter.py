import random

import numpy as np


class BatchIter:
    def __init__(self, train_ids, load_x, load_y, augm_fn, batch_size=32):
        """Creates batch iterator both for keras and torch models.

        Parameters
        ----------
        train_ids: list
            ids to generate train batches.

        load_x: Callable
            `dataset` function to load images.

        load_y: Callable
            `dataset` function to load masks.

        augm_fn: Callable
            Function to augment x/y pairs.

        batch_size: int, optional
            Size of generated batch.
        """
        self.train_ids = train_ids

        self.size = load_x(train_ids[0]).shape

        self.batch_size = batch_size

        x_stack = []
        y_stack = []
        for _id in train_ids:
            x_stack.append(load_x(_id))
            y_stack.append(load_y(_id))
        self.x_stack = np.array(x_stack, dtype='float32')
        self.y_stack = np.array(y_stack, dtype='float32')

        self.augm_fn = augm_fn

        self.inner_ids = np.arange(len(train_ids))

    def flow(self):
        """Generator."""
        batch_x = np.array([np.zeros(self.size)] * self.batch_size, dtype='float32')
        batch_y = np.array([np.zeros(self.size)] * self.batch_size, dtype='float32')

        while True:
            for i in range(self.batch_size):
                index = random.choice(self.inner_ids)
                x = self.x_stack[index]
                y = self.y_stack[index]
                batch_x[i], batch_y[i] = self.augm_fn(x, y)
            yield batch_x, batch_y
