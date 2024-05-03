import numpy as np
import torch
from tqdm import tqdm


def fit_model(nn_model, X, Y, optimizer, loss_func,
              batch_size, epochs, max_steps=1e10):
    """
    :param nn_model:        pytorch model of form Y = f(*X) (class)
    :param X:               tuple of necessary inputs to the function
    :param Y:               desired output from the function (tensor)
    :param optimizer:       optimizer to use
    :param loss_func:       loss criterion
    :param batch_size:      mini-batch size
    :param epochs:          number of epochs
    :return:
    """

    assert type(X) == tuple
    for d in X: assert type(d) == torch.Tensor
    assert type(Y) == torch.Tensor
    device = Y.device
    for d in X: assert d.device == device

    num_samples = Y.shape[0]
    epoch_losses = []
    steps_so_far = 0
    for ep in tqdm(range(epochs)):
        rand_idx = torch.LongTensor(np.random.permutation(num_samples)).to(device)
        ep_loss = 0.0
        num_steps = int(num_samples // batch_size)
        for mb in range(num_steps):
            data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
            batch_X  = [d[data_idx] for d in X]
            batch_Y  = Y[data_idx]
            optimizer.zero_grad()
            Y_hat    = nn_model.forward(*batch_X)
            loss = loss_func(Y_hat, batch_Y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.to('cpu').data.numpy()
        epoch_losses.append(ep_loss * 1.0/num_steps)
        steps_so_far += num_steps
        if steps_so_far >= max_steps:
            print("Number of grad steps exceeded threshold. Terminating early..")
            break
    return epoch_losses