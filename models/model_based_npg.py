from typing import List
import numpy as np

from util.trajectories import Trajectory, compute_advantages
from models.npg_cg import NPG


class ModelBasedNPG(NPG):
    def __init__(self, **kwargs):
        super(ModelBasedNPG, self).__init__(**kwargs)

    def train_step(self, trajectories: List[Trajectory], gamma=0.995, gae_lambda=0.97):
        # NOTE: in the mjrl implementation they removed trajectories that are too short and also they had multiple models for the environment (if they diverged too much, they'd truncate the trajectories)
    
        # train from paths
        returns = [trajectory.compute_discounted_rewards(gamma) for trajectory in trajectories]
        advantages = compute_advantages(trajectories, self.baseline, gamma, gae_lambda)

        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(len(trajectories))

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
