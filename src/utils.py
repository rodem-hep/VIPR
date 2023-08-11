import numpy as np
import torch as T
import io
from PIL import Image

def log_squash(data: np.ndarray) -> np.ndarray:
    """Apply a log squashing function for distributions with high tails."""
    if isinstance(data, T.Tensor):
        return T.sign(data) * T.log(T.abs(data) + 1)
    else:
        return np.sign(data) * np.log(np.abs(data) + 1)


def undo_log_squash(data: np.ndarray) -> np.ndarray:
    """Undo the log squash function above."""
    if isinstance(data, T.Tensor):
        return T.sign(data) * (T.exp(T.abs(data)) - 1)
    else:
        return np.sign(data) * (np.exp(np.abs(data)) - 1)


def append_dims(x: T.Tensor, target_dims: int, add_to_front: bool = False) -> T.Tensor:
    """Appends dimensions of size 1 to the end or front of a tensor until it
    has target_dims dimensions.

    Parameters
    ----------
    x : T.Tensor
        The input tensor to be reshaped.
    target_dims : int
        The target number of dimensions for the output tensor.
    add_to_front : bool, optional
        If True, dimensions are added to the front of the tensor.
        If False, dimensions are added to the end of the tensor.
        Defaults to False.

    Returns
    -------
    T.Tensor
        The reshaped tensor with target_dims dimensions.

    Raises
    ------
    ValueError
        If the input tensor already has more dimensions than target_dims.

    Examples
    --------
    >>> x = T.tensor([1, 2, 3])
    >>> x.shape
    torch.Size([3])

    >>> append_dims(x, 3)
    tensor([[[1]], [[2]], [[3]]])
    >>> append_dims(x, 3).shape
    torch.Size([3, 1, 1])

    >>> append_dims(x, 3, add_to_front=True)
    tensor([[[[1, 2, 3]]]])
    >>> append_dims(x, 3, add_to_front=True).shape
    torch.Size([1, 1, 3])
    """
    dim_diff = target_dims - x.dim()
    if dim_diff < 0:
        raise ValueError(f"x has more dims ({x.ndim}) than target ({target_dims})")
    if add_to_front:
        return x[(None,) * dim_diff + (...,)]  # x.view(*dim_diff * (1,), *x.shape)
    return x[(...,) + (None,) * dim_diff]  # x.view(*x.shape, *dim_diff * (1,))


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img