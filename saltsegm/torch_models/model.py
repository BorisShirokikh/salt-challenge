import torch
from tqdm import tqdm

from .torch_utils import to_var, to_np, calc_val_metric, logits2pred


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
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

        # change to set_optimizer
        self.optimizer = optim(self.model.parameters())

        if not lr_scheduler is None:
            self.lr_scheduler = lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None

    def do_train_step(self, x, y):
        """Model performs single train step."""
        self.model.train()

        x_t = to_var(x)
        y_t = to_var(y, requires_grad=False)

        pred = self.model(x_t)
        loss = self.loss_fn(pred, y_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(to_np(loss))

    def do_val_step(self, x, y):
        """Model performs signle validation step."""
        self.model.eval()

        x_t = to_var(x, requires_grad=False)
        y_t = to_var(y, requires_grad=False)

        with torch.no_grad():
            logit = self.model(x_t)

        loss = self.loss_fn(logit, y_t)

        pred = logits2pred(logit)

        metric = calc_val_metric(y_t, pred, self.metric_fn)

        return float(to_np(loss)), metric

    def do_inf_step(self, x):
        """Model preforms single inference step."""
        self.model.eval()

        x_t = to_var(x, requires_grad=False)

        with torch.no_grad():
            logit = self.model(x_t)

        pred = logits2pred(logit)

        return to_np(pred)

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

                if not val_data is None:
                    l, m = self.do_val_step(val_data[0], val_data[1])

                    val_losses.append(l)
                    val_metrics.append(m)

                    pbar.set_postfix(val_loss=l, val_metric=m)
                    pbar.update()
                    
                    if not self.lr_scheduler is None:
                        self.lr_scheduler.step(l)
                        
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
