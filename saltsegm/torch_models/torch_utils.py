import numpy as np
import torch
from torch.autograd import Variable


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
    x = Variable(torch.from_numpy(x),
                 requires_grad=requires_grad, volatile=not requires_grad)
    return to_cuda(x, cuda)


def logits2pred(logit):
    """Transforms logit output to probability-like output."""
    return logit.exp() / (logit.exp() + 1)
