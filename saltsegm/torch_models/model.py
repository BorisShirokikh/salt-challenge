import os
from math import inf

import torch
from tqdm import tqdm

from saltsegm.metrics import calc_val_metric
from saltsegm.utils import is_better, dump_json, get_pred, get_spatial
from .torch_utils import to_np, to_var, logits2pred  # TODO: change to_var()


def do_train_step(x, y, model, optimizer, loss_fn):
    model.train()

    x = to_var(x)
    y = to_var(y, requires_grad=False)

    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_to_return = float(loss.item())

    return loss_to_return


def do_inf_step(x, model):
    with torch.no_grad():
        model.eval()
        x = to_var(x, requires_grad=False)
        pred = to_np(logits2pred(model(x)))

    return pred


def do_val_step(x, y, model, loss_fn, metric_fn, pred_fn):
    with torch.no_grad():
        model.eval()
        x = to_var(x, requires_grad=False)
        y = to_var(y, requires_grad=False)

        logits = model(x)
        y_pred = logits2pred(logits)

        loss = loss_fn(logits, y)
        loss_to_return = float(loss.item())

    metric = calc_val_metric(y, y_pred, metric_fn, pred_fn)

    del x, y, loss
    torch.cuda.empty_cache()

    return loss_to_return, metric


class TorchModel:
    def __init__(self, model, loss_fn, metric_fn, optim, lr_scheduler=None, task_type='segm'):
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

        task_type: str, optional
            `segm` or `other` type of task.
        """
        self.model = model

        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

        self.optim = optim
        self.optimizer = optim(self.model.parameters())

        self.lr_scheduler_fn = lr_scheduler

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None

        assert task_type in ('segm', 'other', ), \
            f'`task_type` should be `segm` or `other`, {task_type} given'
        self.task_type = task_type

    def to_cuda(self):
        self.model.cuda()
        self.loss_fn.cuda()
        self.optimizer = self.optim(self.model.parameters())
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.lr_scheduler_fn(self.optimizer)

    def do_train_step(self, x, y):
        """Model performs single train step."""
        return do_train_step(x, y, self.model, self.optimizer, self.loss_fn)

    def do_val_step(self, x, y):
        """Model performs single validation step."""
        if self.task_type == 'segm':
            pred_fn = get_pred
        else:
            pred_fn = get_spatial

        return do_val_step(x, y, self.model, self.loss_fn, self.metric_fn, pred_fn=pred_fn)

    def do_inf_step(self, x):
        """Model preforms single inference step."""
        return do_inf_step(x, self.model)


def fit_model(torch_model, generator, val_path, val_data=None, epochs=2, steps_per_epoch=100, verbose=True,
              saving_model_mode='max', use_cuda=True):
    """Function to fit model from generator.

    Parameters
    ----------
    torch_model: TorchModel
        Initialized model to fit.

    generator: generator
        Object to generate training batches.

    epochs: int, optional
        Number of epochs to train.

    val_path: str
        Path to save model.

    val_data: tuple, None, optional
        Data `(x, y)` to perform validation steps. If `None`, will skip val steps.

    steps_per_epoch: int, optional
        Steps model makes per one epoch.

    verbose: bool, optional
        If `True`, will show the online progress of training process.
        If `False`, will be silent.

    saving_model_mode: str, None, optional
        If `None`, will save model after the last epoch.
        If `min`, will save the latest model with the lowest loss after val step.
        If `max`, will save the latest model with the highest metric after val step.
        
    use_cuda: bool, optional
        If `True`, will transfer model to cuda device.
    """
    # *** Init stage ***
    if val_data is None:
        assert torch_model.lr_scheduler is None, 'LR scheduler cannot be used without val data'

    if saving_model_mode is not None:
        assert saving_model_mode in ('min', 'max'), 'saving_model_mode should be `min` or `max`'

    if use_cuda:
        torch_model.to_cuda()

    best = None
    if saving_model_mode == 'min':
        best = inf
    elif saving_model_mode == 'max':
        best = -inf

    train_losses = []
    val_losses, val_metrics, val_lrs = [], [], []

    # *** train-val stage ***
    for n_ep in range(epochs):

        pbar_div = 100
        with tqdm(desc=f'epoch {n_ep+1}/{epochs}', total=steps_per_epoch,
                  unit_divisor=steps_per_epoch // pbar_div, disable=not verbose,
                  ) as pbar:

            # *** TRAIN STEPS ***
            for n_step in range(steps_per_epoch):
                x_batch, y_batch = next(generator)

                l = torch_model.do_train_step(x_batch, y_batch)
                train_losses.append(l)

                if n_step % 10 == 0:
                    pbar.set_postfix(train_loss=l)
                pbar.update()
            # end for
            torch.cuda.empty_cache()

            # *** VAL STEP ***
            if val_data is not None:
                l, m = torch_model.do_val_step(val_data[0], val_data[1])

                val_losses.append(l)
                val_metrics.append(m)

                if torch_model.lr_scheduler is not None:
                    torch_model.lr_scheduler.step(l, epoch=n_ep)

                lr = torch_model.optimizer.param_groups[0]['lr']
                val_lrs.append(lr)

                pbar.set_postfix(val_loss=l, val_metric=m, lr=lr)
                pbar.update()

                # *** SAVING MODEL ***
                cur = None
                if saving_model_mode == 'min':
                    cur = l
                elif saving_model_mode == 'max':
                    cur = m

                if saving_model_mode is not None:
                    if is_better(cur=cur, best=best, mode=saving_model_mode):
                        torch.save(torch_model.model, os.path.join(val_path, 'model.pt'))
    # end for

    if saving_model_mode is None:
        torch.save(torch_model.model, os.path.join(val_path, 'model.pt'))

    # *** SAVING LOG ***
    if val_data is None:
        history = {'train_losses': train_losses}
    else:
        history = {'train_losses': train_losses,
                   'val_losses': val_losses,
                   'val_metrics': val_metrics,
                   'val_lrs': val_lrs}

    dump_json(history, os.path.join(val_path, 'log.json'))
