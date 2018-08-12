import numpy as np
import torch

from saltsegm.utils import get_pred


def to_cuda(x, cuda: bool = None):
    if cuda or (cuda is None and torch.cuda.is_available()):
        x = x.cuda()
    return x


def to_np(x: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a numpy array."""
    return x.data.cpu().numpy()


def to_var(x: np.ndarray, cuda: bool = None, requires_grad: bool = True) -> torch.Tensor:
    """
    Convert a numpy array to a torch Tensor
    Parameters
    ----------
    x: np.ndarray
    cuda: bool, optional
        move tensor to cuda. If None, torch.cuda.is_available() is used to determine that.
    requires_grad: bool, optional
    """
    x = torch.from_numpy(np.asarray(x))
    if requires_grad:
        x.requires_grad_()
    return to_cuda(x, cuda)


def calc_val_metric(true_t, pred_t, metric_fn):
    """Calculates metric `metric_fn` during the validation step.
    
    Parameters
    ----------
    true_t: torch.DoubleTensor, torch.cuda.DoubleTensor
        torch tensor corresponding to ground truth.
        
    pred_t: torch.DoubleTensor, torch.cuda.DoubleTensor
        torch tensor corresponding to prediction.
        
    metric_fn: Callable
        Function to calculate metric between two numpy tensors.
    """
    true_np = to_np(true_t)
    pred_np = to_np(pred_t)

    metric_list = []
    for t, p in zip(true_np, pred_np):
        metric_list.append(metric_fn(get_pred(t), get_pred(p)))

    return np.mean(metric_list)


def logits2pred(logit):
    """Transforms logit output to probability-like output."""
    return logit.exp() / (logit.exp() + 1)
