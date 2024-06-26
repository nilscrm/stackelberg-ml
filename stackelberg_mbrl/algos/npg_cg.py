# NOTE: stripped it down to what we need, see original file if additional functionality needed

import logging
from typing import List
import numpy as np
import time as timer
import torch

from stackelberg_mbrl.algos.batch_reinforce import BatchREINFORCE
from stackelberg_mbrl.policies.policy import ATrainablePolicy
from stackelberg_mbrl.util.trajectories import TrajectoryList
from stackelberg_mbrl.util.cg_solve import cg_solve

logging.disable(logging.CRITICAL)


class NPG(BatchREINFORCE):
    def __init__(self, policy: ATrainablePolicy,
                 normalized_step_size=0.01,
                 const_learn_rate=None,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 save_logs=False,
                 kl_dist=None,
                 ):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_dist: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        """
        super().__init__(policy, save_logs)

        self.alpha = const_learn_rate
        self.n_step_size = normalized_step_size if kl_dist is None else 2.0 * kl_dist
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None

    def HVP(self, observations, actions, vec, regu_coef=None, device=None):
        """ Compute the Hessian-vector product """
        regu_coef = self.FIM_invert_args['damping'] if regu_coef is None else regu_coef
        device = self.policy.device if device is None else device
        assert type(vec) == torch.Tensor
        assert type(regu_coef) == float
        vec = vec.to(device)
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            rand_idx = np.random.choice(num_samples, size=int(self.hvp_subsample*num_samples))
            observations = observations[rand_idx]
            actions = actions[rand_idx]
        kl_div = self.policy.kl_divergence(observations, actions)
        grad_fo = torch.autograd.grad(kl_div, self.policy.trainable_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
        gvp = torch.sum(flat_grad*vec)
        hvp = torch.autograd.grad(gvp, self.policy.trainable_params)
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat + regu_coef*vec

    def build_Hvp_eval(self, inputs, regu_coef=None, device=None):
        def eval(v):
            full_inp = inputs + [v] + [regu_coef] + [device]
            Hvp = self.HVP(*full_inp)
            return Hvp
        return eval

    # ----------------------------------------------------------
    def train_on_trajectories(self, trajectories: TrajectoryList, advantages: List[np.ndarray]):
        states = np.concatenate(trajectories.states)
        actions = np.concatenate(trajectories.actions)
        advantages = np.concatenate(advantages)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        if self.save_logs:
            pg_surr = self.pg_surrogate(states, actions, advantages)
            surr_before = pg_surr.to('cpu').data.numpy().ravel()[0]
            actions_before = self.policy.sample_next_actions(states).float()

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(states, actions, advantages)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([states, actions],
                                  regu_coef=self.FIM_invert_args['damping'],
                                  device=vpg_grad.device)
        npg_grad = cg_solve(f_Ax=hvp, b=vpg_grad, x_0=vpg_grad.clone(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha ** 2) * vpg_grad.dot(npg_grad)
        else:
            n_step_size = self.n_step_size
            inner_prod = vpg_grad.dot(npg_grad)
            alpha = torch.sqrt(torch.abs(self.n_step_size / (inner_prod + 1e-10)))
            alpha = alpha.to('cpu').data.numpy().ravel()[0]

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params.clone())

        # Log information
        if self.save_logs:
            total_rewards = trajectories.total_rewards

            mean_return = np.mean(total_rewards)
            std_return = np.std(total_rewards)
            min_return = np.amin(total_rewards)
            max_return = np.amax(total_rewards)

            if self.running_score is None:
                self.running_score = mean_return
            else:
                self.running_score = 0.9 * self.running_score + 0.1 * mean_return
                
            pg_surr = self.pg_surrogate(states, actions, advantages)
            surr_after = pg_surr.to('cpu').data.numpy().ravel()[0]
            kl_divergence = self.policy.kl_divergence(states, actions_before).item()

            self.logger.log_kv('stoc_pol_mean', mean_return)
            self.logger.log_kv('stoc_pol_std', std_return)
            self.logger.log_kv('stoc_pol_max', max_return)
            self.logger.log_kv('stoc_pol_min', min_return)

            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('kl_dist', kl_divergence)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)

        return dict(
            num_trajectories = trajectories.num_trajectories,
            mean_return = mean_return, 
            std_return = std_return, 
            min_return = min_return, 
            max_return = max_return
        )