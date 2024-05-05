import torch
import torch.nn.functional as F
import numpy as np
from functools import wraps

def tensorize_array_inputs(func):
    """ Decorator that tensorizes any numpy array inputs to the function """
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                args[i] = tensorize(arg)
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                kwargs[key] = tensorize(arg)
        return func(*args, **kwargs)
    return wrapper

def extract_one_hot_index_inputs(func):
    """ Decorator that extracts the argmax from any numpy array or tensor input """
    # NOTE: only works if all torch tensors or numpy arrays are one hot!
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray) or isinstance(arg, torch.Tensor):
                args[i] = one_hot_to_idx(arg)
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                kwargs[key] = one_hot_to_idx(arg)
        return func(*args, **kwargs)
    return wrapper

def one_hot_to_idx(one_hot):
    return np.argmax(one_hot).item()

def one_hot(idx: int, num_classes: int):
    """ One hot encoding of single integer value, will give a single tensor (no batching) """
    return F.one_hot(torch.LongTensor([idx]), num_classes)[0]

def tensorize(var, device='cpu'):
    """
    Convert input to torch.Tensor on desired device
    :param var: type either torch.Tensor or np.ndarray
    :param device: desired device for output (e.g. cpu, cuda)
    :return: torch.Tensor mapped to the device
    """
    if type(var) == torch.Tensor:
        return var.float().to(device)
    elif type(var) == np.ndarray:
        return torch.from_numpy(var).float().to(device)
    elif type(var) == float:
        return torch.tensor(var).float()
    else:
        print("Variable type not compatible with function.")
        return None