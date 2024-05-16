import numpy as np
import torch

def fit_tuple(model, X, Y, optimizer, loss_func,
              batch_size, epochs, max_steps=1e10, visualizer: callable = (lambda x: x)):
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

    num_samples = Y.shape[0]
    epoch_losses = []
    steps_so_far = 0
    for ep in visualizer(range(epochs)):
        rand_idx = torch.LongTensor(np.random.permutation(num_samples))
        ep_loss = 0.0
        num_steps = int(num_samples // batch_size)
        for mb in range(num_steps):
            data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
            batch_X  = [d[data_idx] for d in X]
            batch_Y  = Y[data_idx]
            optimizer.zero_grad()
            Y_hat = model.forward(*batch_X)
            loss = loss_func(Y_hat.squeeze(), batch_Y.squeeze())
            loss.backward()
            optimizer.step()
            ep_loss += loss.to('cpu').data.numpy()
        epoch_losses.append(ep_loss * 1.0/num_steps)
        steps_so_far += num_steps
        if steps_so_far >= max_steps:
            print("Number of grad steps exceeded threshold. Terminating early..")
            break
    return epoch_losses

def fit(model, X, Y, optimizer, loss_func,
        batch_size, epochs, max_steps=1e10, visualizer: callable = (lambda x: x)):
    """ Fit a model such that model(X) = Y """
    return fit_tuple(model, [X], Y, optimizer, loss_func, batch_size, epochs, max_steps, visualizer)