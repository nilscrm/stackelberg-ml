import torch
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
                args[i] = np.argmax(arg).item()
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                kwargs[key] = np.argmax(arg).item()
        return func(*args, **kwargs)
    return wrapper

def tensorize(var, device='cpu'):
    """
    Convert input to torch.Tensor on desired device
    :param var: type either torch.Tensor or np.ndarray
    :param device: desired device for output (e.g. cpu, cuda)
    :return: torch.Tensor mapped to the device
    """
    if type(var) == torch.Tensor:
        return var.to(device)
    elif type(var) == np.ndarray:
        return torch.from_numpy(var).float().to(device)
    elif type(var) == float:
        return torch.tensor(var).float()
    else:
        print("Variable type not compatible with function.")
        return None