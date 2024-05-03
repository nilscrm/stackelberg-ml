# NOTE: stripped it down to what we need, see original file if additional functionality needed

"""
Basic reinforce algorithm using on-policy rollouts
Also has function to perform linesearch on KL (improves stability)
"""

import logging
logging.disable(logging.CRITICAL)
import torch
from torch.autograd import Variable
from util.logger import DataLog


class BatchREINFORCE:
    def __init__(self, policy, save_logs=False):
        self.policy = policy
        self.save_logs = save_logs
        if save_logs: self.logger = DataLog()

    def pg_surrogate(self, observations, actions, advantages):
        # grad of the surrogate is equal to the REINFORCE gradient
        # need to perform ascent on this objective function
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        mean, LL = self.policy.mean_LL(observations, actions)
        adv_var = adv_var.to(LL.device)
        surr = torch.mean(LL*adv_var)
        return surr

    def kl_old_new(self, observations, old_mean, old_log_std, *args, **kwargs):
        new_mean = self.policy.forward(observations)
        new_log_std = self.policy.log_std
        kl_divergence = self.policy.kl_divergence(new_mean, old_mean, new_log_std,
                                                  old_log_std, *args, **kwargs)
        return kl_divergence.to('cpu').data.numpy().ravel()[0]

    def flat_vpg(self, observations, actions, advantages):
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(pg_surr, self.policy.trainable_params)
        return torch.cat([g.contiguous().view(-1) for g in vpg_grad])
