import torch
import torch.nn as nn
from tqdm import tqdm

from .torch_utils import to_var, to_np, calc_val_metric


class TorchModel:
    def __init__(self, model, loss_fn, metric_fn, optim, lr):
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optim(self.model.parameters(), lr=lr)

    def do_train_step(self, x, y):
        x_t = to_var(x)
        y_t = to_var(y, requires_grad=False)

        pred = self.model(x_t)
        loss = self.loss_fn(pred, y_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return to_np(loss)

    def do_val_step(self, x, y):
        x_t = to_var(x, requires_grad=False)
        y_t = to_var(y, requires_grad=False)

        with torch.no_grad():
            pred = self.model(x_t)

        loss = self.loss_fn(pred, y_t)
        metric = calc_val_metric(y_t, pred, self.metric_fn)

        return to_np(loss), metric

    def do_inf_step(self, x):
        x_t = to_var(x)

        with torch.no_grad():
            pred = self.model(x_t)

        return to_np(pred)

    def fit_generator(self, generator, epochs=2, val_data=None,
                      steps_per_epoch=100, verbose=True):
        train_losses = []
        val_losses, val_metrics = [], []

        for n_ep in range(epochs):

            pbar_div = 100
            with tqdm(desc=f'epoch {n_ep+1}/{epochs}', total=steps_per_epoch,
                      unit_divisor=steps_per_epoch // pbar_div, disable=not verbose,
                      ) as pbar:

                for n_step in range(steps_per_epoch):
                    x_batch, y_batch = next(generator)

                    l = self.do_train_step(x_batch, y_batch)
                    train_losses.append(l)

                    if n_step % 10 == 0:
                        pbar.set_postfix(train_loss=l)
                    pbar.update()
                # end for

                if not val_data is None:
                    l, m = self.do_val_step(val_data[0], val_data[1])

                    val_losses.append(l)
                    val_metrics.append(m)

                    pbar.set_postfix(val_loss=l, val_metric=m)
                    pbar.update()
        # end for  

        if val_data is None:
            return {'train_losses': train_losses}
        else:
            return {'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_metrics': val_metrics}
