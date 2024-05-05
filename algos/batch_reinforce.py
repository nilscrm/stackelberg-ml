# NOTE: stripped it down to what we need, see original file if additional functionality needed

"""
Basic reinforce algorithm using on-policy rollouts
Also has function to perform linesearch on KL (improves stability)
"""

import logging

from policies.policy import APolicy
logging.disable(logging.CRITICAL)
import torch
from torch.autograd import Variable
from util.logger import DataLog


class BatchREINFORCE:
    def __init__(self, policy: APolicy, save_logs=False):
        self.policy = policy
        self.save_logs = save_logs
        if save_logs: self.logger = DataLog()

    def pg_surrogate(self, observations, actions, advantages):
        # grad of the surrogate is equal to the REINFORCE gradient
        # need to perform ascent on this objective function
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        LL = self.policy.log_likelihood(observations, actions)
        return torch.mean(LL*adv_var)

    def flat_vpg(self, observations, actions, advantages):
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(pg_surr, self.policy.trainable_params)
        return torch.cat([g.contiguous().view(-1) for g in vpg_grad])
