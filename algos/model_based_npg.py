# NOTE: stripped it down to what we need, see original file if additional functionality needed
import numpy as np

from nn.baseline.baselines import ABaseline
from util.trajectories import TrajectoryList
from algos.npg_cg import NPG

class ModelBasedNPG(NPG):
    def train_step(self, trajectories: TrajectoryList, baseline: ABaseline, gamma=0.995, gae_lambda=0.97):
        # NOTE: in the mjrl implementation they removed trajectories that are too short and also they had multiple models for the environment (if they diverged too much, they'd truncate the trajectories)
        # NOTE: we compute the advantages before the baseline update to avoid overfitting to the current trajectories
        observations = np.concatenate(trajectories.states)
        returns = trajectories.compute_discounted_rewards(gamma)
        advantages = trajectories.compute_advantages(baseline, gamma, gae_lambda, normalize=False)
        eval_statistics = self.train_on_trajectories(trajectories, advantages)

        # fit baseline
        if self.save_logs:
            error_before, error_after = baseline.fit(observations, returns, return_errors=True)
            self.logger.log_kv('num_samples', trajectories.num_samples)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            baseline.fit(observations, returns)

        return eval_statistics
