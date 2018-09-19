import torch
from tqdm import tqdm

from saltsegm.metrics import calc_val_metric
from .torch_utils import to_var, to_np, logits2pred


def do_train_step(x, y, model, optimizer, loss_fn):
    model.train()

    x_t = to_var(x)
    y_t = to_var(y, requires_grad=False)

    logits = model(x_t)
    loss = loss_fn(logits, y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_to_return = float(to_np(loss))

    del model, x_t, y_t, logits, loss, optimizer

    return loss_to_return


def do_inf_step(x, model):
    with torch.no_grad():
        model.eval()
        x_t = to_var(x, requires_grad=False)
        pred = to_np(logits2pred(model(x_t)))

    del x_t, model

    return pred


def do_val_step(x, y, model, loss_fn, metric_fn):
    with torch.no_grad():
        model.eval()
        x_t = to_var(x, requires_grad=False)
        y_t = to_var(y, requires_grad=False)

        logits = model(x_t)
        y_pred = logits2pred(logits)

        loss = loss_fn(logits, y_t)
        loss_to_return = float(to_np(loss))

    metric = calc_val_metric(to_np(y_t), to_np(y_pred), metric_fn)

    del model, x_t, y_t, logits, y_pred, loss

    return loss_to_return, metric


class TorchModel:
    def __init__(self, model, loss_fn, metric_fn, optim, lr_scheduler=None):
        """Custom torch model class to handle basic operations.

        Parameters
        ----------
        model: torch.nn.Module, or the same
            Model graph.

        loss_fn: torch.nn.modules.loss, or the same
            Loss function to calculate gradients.

        metric_fn: Callable
            Function to calculate metric between `true` and `pred` tensors
            during the validation step.

        optim: torch.optim, or the same
            Optimizer to do back propagation step.

        lr_scheduler: torch.optim.lr_scheduler, or the same
            Scheduler to control learning rate changing during the training.
        """
        self.model = model.cuda()

        self.loss_fn = loss_fn.cuda()
        self.metric_fn = metric_fn

        # change to set_optimizer
        self.optimizer = optim(self.model.parameters())

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None

    def do_train_step(self, x, y):
        """Model performs single train step."""
        return do_train_step(x, y, self.model, self.optimizer, self.loss_fn)

    def do_val_step(self, x, y):
        """Model performs single validation step."""
        return do_val_step(x, y, self.model, self.loss_fn, self.metric_fn)

    def do_inf_step(self, x):
        """Model preforms single inference step."""
        return do_inf_step(x, self.model)

    def fit_generator(self, generator, epochs=2, val_data=None,
                      steps_per_epoch=100, verbose=True):
        """Function to fit model from generator.

        Parameters
        ----------
        generator: generator
            Object to generate training batches.

        epochs: int, optional
            Number of epochs to train.

        val_data: tuple, None, optional
            Data `(x, y)` to perform validation steps. If `None`, will skip val steps.
            
        steps_per_epoch: int, optional
            Steps model makes per one epoch.

        verbose: bool
            If `True`, will show the online progress of training process.
            If `False`, will be silent.
            
        Returns
        -------
        history: dict
            dict with stored losses and metrics (if `val_data` is not `None`) values.
        """
        if val_data is None:
            assert self.lr_scheduler is None, 'LR scheduler cannot be used without val data'

        train_losses = []
        val_losses, val_metrics, val_lrs = [], [], []

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

                if val_data is not None:
                    with torch.no_grad():
                        l, m = self.do_val_step(val_data[0], val_data[1])
                        self.optimizer.zero_grad()

                    val_losses.append(l)
                    val_metrics.append(m)

                    pbar.set_postfix(val_loss=l, val_metric=m)
                    pbar.update()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step(l, epoch=n_ep+1)

                    lr = self.optimizer.param_groups[0]['lr']
                    val_lrs.append(lr)
        # end for  

        if val_data is None:
            return {'train_losses': train_losses}
        else:
            return {'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_metrics': val_metrics,
                    'val_lrs': val_lrs}
