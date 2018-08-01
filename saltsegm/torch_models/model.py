import torch
import torch.nn as nn
from tqdm import tqdm

from .torch_utils import to_var, to_np


class TorchModel:
    def __init__(self, model, loss_fn, optim, lr):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optim(self.model.parameters(), lr=lr)


    def do_train_step(self, x, y):
        x_t = to_var(x)
        y_t = to_var(y, requires_grad=False)

        pred = self.model(x_t)

        loss = self.loss_fn(pred, y_t)

        self.optimizer.zero_grad()

        loss.backward()
        
        self.optimizer.step()

        return loss
    
    
    def do_val_step(self):
        return 0

    
    def do_inf_step(self, x):
        x_t = to_var(x)

        with torch.no_grad():
            pred = self.model(x_t)

        return to_np(pred)
    
    
    def fit_generator(self, generator, epochs=2, val_data=None,
                      steps_per_epoch=100):
        train_losses = []
        val_metrics = []
        for n_ep in tqdm(range(epochs)):
            for n_step in range(steps_per_epoch):
                x_batch, y_batch = next(generator)
                
                l = self.do_train_step(x_batch, y_batch)
                train_losses.append(l)

            m = self.do_val_step()
            val_metrics.append(m)

        return train_losses, val_metrics
