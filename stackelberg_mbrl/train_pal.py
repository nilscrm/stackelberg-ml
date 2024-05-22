from itertools import product
import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stackelberg_mbrl.nn.model.world_models import ContextualizedWorldModel
from stackelberg_mbrl.policies.random_policy import RandomPolicy
import torch
import gymnasium

import stackelberg_mbrl.envs.simple_mdp
from stackelberg_mbrl.envs.querying_env import LeaderEnv, ModelQueryingEnv, PolicyQueryingEnv
from stackelberg_mbrl.envs.env_util import transition_probabilities_from_world_model, draw_mdp, RandomMDP, LearnableWorldModel
from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, LoadPolicy, PolicyConfig, LoadWorldModel, WorldModelConfig
from stackelberg_mbrl.experiments.model_rl.config import model_rl_config

from stackelberg_mbrl.util.trajectories import sample_trajectories

def train_contextualized_PAL(config: ExperimentConfig):
    """
    In contextualized PAL we condition and pretrain the model on random policies.
    This way we get an oracle that behaves like the best model conditioned on each policy.
    In doing so, we can then optimize for an optimal policy-model pair.

    """
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    # Groundtruth environment, which we sample from
    real_env = gymnasium.make(config.env_config.env_true_id, max_ep_steps=config.env_config.max_episode_steps)
    queries = list(range(real_env.num_states))

    random_policy = RandomPolicy(real_env.num_states, real_env.num_actions)

    # Pretrain the model conditioned on a policy
    contextualized_real_env = PolicyQueryingEnv(
        env=real_env, 
        policy=random_policy, 
        queries=queries, 
        on_reset=random_policy.randomize)
    contextualized_model = ContextualizedWorldModel(len(queries)*real_env.num_actions, ...)

    print("Pretraining world model")
    for iter in range(pretrain_iterations):
        # TODO: can potentially re-use samples from real-env?
        # Does some pretraining of the model oracle under random policies
        
        # generate rollouts on the real environment (using the current policy)
        trajectories = sample_trajectories(contextualized_real_env, random_policy, num_trajectories, max_steps)

        s = trajectories.states
        a = trajectories.actions
        r = trajectories.rewards
        s_next = trajectories.next_states

        # use sgd to match the transitions
        contextualized_model.fit_dynamics(s, a, s_next) # use CE-loss (but only on state, not the context!)

        if learn_reward:
            contextualized_model.fit_reward(s, a, s_next, r)


        # TODO: SGD on samples from model under random policies
        # for any given policy, we only need the model to be accurate in predicting s_next of (s,a) that actually occur

        # context = (a_distr) forall s

        # if env state was prepended with context (queried from policy):
        # + env could call randomize on policy reference whenever it is reset
        # - pretraining would happen on (context | s), a_oh -> (context | s_next) [could extract s_next though => (context | s), a_oh -> s_next, the world model could also extract and re-append the exact context for consistency]
        # - during training, env would receive a_oh as inputs

        # TODO: Implement
        # - randomizable policy
        # - query environment that queries policy and with randomize_on_reset flag (maybe use general follower interface to allow for code reuse with MAL)
        # - contextualized world model (receives as input the context and state, extracts the context to re-append it at the end)
        # - sampling (should work out of the box)
        # - training loop (use CE loss!)




        # NOTE: shouldn implement it like this bc it will make things complicated with the custom policy
        # if action was appended with context (queried from policy):
        # - manually randomize the policy [e.g. after each rollout]
        # + pretraining would happen on s, (a_oh | context) -> s_next
        # - would require custom policy network [that appends any action with context. can stable baselines deal with oh-actions? - depends on env I guess]
        # + during training, env would receive (a_oh | context) as inputs

        

    # Train the policy with the best-responding models
    print("Training policy")
    # env where the first steps are the s from the queries (and 0 reward is given) 
    # before it uses the pretrained best responding model to simulate the world
    contextualized_leader_env = PolicyQueryingEnv(
        env=SimpleLeaderEnv(contextualized_model, ), 
        queries=queries)

    policy_config: PolicyConfig = config.policy_config
    # TODO: need custom PPO implementation that has callback before doing rollouts (so we can load the oracle weights and finetune it to the current policy)
    policy_ppo = PPO( 
        "MlpPolicy",
        contextualized_leader_env,
        policy_kwargs=policy_config.policy_kwargs,
        tensorboard_log=config.output_dir / config.experiment_name / "tb",
    )
    

    # I hope their PPO can handle (context, s) as inputs to the environment...
    policy_ppo.learn(samples_per_training_iteration, tb_log_name="Policy", reset_num_timesteps=False)


if __name__ == "__main__":
    train_contextualized_PAL(model_rl_config)
