# NOTE: stripped it down to what we need, see original file if additional functionality needed

import numpy as np

from util.trajectories import TrajectoryList
from algos.npg_cg import NPG


class ModelBasedNPG(NPG):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_step(self, trajectories: TrajectoryList, gamma=0.995, gae_lambda=0.97):
        # NOTE: in the mjrl implementation they removed trajectories that are too short and also they had multiple models for the environment (if they diverged too much, they'd truncate the trajectories)
    
        advantages = trajectories.compute_advantages(self.baseline, gamma, gae_lambda, normalize=False)
        eval_statistics = self.train_on_trajectories(trajectories, advantages)

        # log number of samples
        if self.save_logs:
            num_samples = np.sum([trajectory.length for trajectory in trajectories])
            self.logger.log_kv('num_samples', num_samples)

        # fit baseline
        if self.save_logs:
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics
