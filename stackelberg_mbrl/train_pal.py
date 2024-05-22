from itertools import product
import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import gymnasium

import stackelberg_mbrl.envs.simple_mdp
from stackelberg_mbrl.envs.querying_env import LeaderEnv, ModelQueryingEnv
from stackelberg_mbrl.envs.env_util import transition_probabilities_from_world_model, draw_mdp, RandomMDP, LearnableWorldModel
from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, LoadPolicy, PolicyConfig, LoadWorldModel, WorldModelConfig
from stackelberg_mbrl.experiments.model_rl.config import model_rl_config


def train_contextualized_PAL(config: ExperimentConfig):
    """
    In contextualized PAL we condition and pretrain the model on random policies.
    This way we get an oracle that behaves like the best model conditioned on each policy.
    In doing so, we can then optimize for an optimal policy-model pair.

    """
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    # Groundtruth environment, which we sample from
    env_true = gymnasium.make(config.env_config.env_true_id, max_ep_steps=config.env_config.max_episode_steps)
    queries = list(range(env_true.num_states))

    # Pretrain the model conditioned on a policy
    print("Pretraining world model")
    for iter in range(pretrain_iterations):
        ...
        # TODO: SGD on samples from model under random policies
        # for any given policy, we only need the model to be accurate in predicting s_next of (s,a) that actually occur

        # context = (a_distr) forall s

        # if env state was prepended with context (queried from policy):
        # + env could call randomize on policy reference whenever it is reset
        # - pretraining would happen on (context | s), a_oh -> (context | s_next) [could extract s_next though => (context | s), a_oh -> s_next, the world model could also extract and re-append the exact context for consistency]
        # - during training, env would receive a_oh as inputs

        # if action was appended with context (queried from policy):
        # - manually randomize the policy [e.g. after each rollout]
        # + pretraining would happen on s, (a_oh | context) -> s_next
        # - would require custom policy network [that appends any action with context. can stable baselines deal with oh-actions? - depends on env I guess]
        # + during training, env would receive (a_oh | context) as inputs

        

    # Train the policy with the best-responding models
    print("Training policy")
    # TODO: PPO against best-responding model
    # how does it see the queries: just like in MAL, the first few steps in the env are s, a_distr, context -> s_next

    # TODO: how to implement inner-loop here?
    # probably need to extend PPO, s.t. we have a callback before each rollout which allows us to load original weights and pretrain [also will have to check if it doesnt use a rollout buffer with samples from old model]
    policy_ppo.learn(samples_per_training_iteration, tb_log_name="Policy", reset_num_timesteps=False)


if __name__ == "__main__":
    train_contextualized_PAL(model_rl_config)
