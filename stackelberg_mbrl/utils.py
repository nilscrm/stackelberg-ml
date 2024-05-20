import numpy as np

def one_hot(x: int, num_elements: int) -> np.ndarray:
    one_hot = np.zeros((num_elements, ))
    one_hot[x] = 1
    return one_hot
