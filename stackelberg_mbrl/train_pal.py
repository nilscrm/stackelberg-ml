from itertools import product
import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stackelberg_mbrl.nn.model.world_models import ContextualizedWorldModel
from stackelberg_mbrl.policies.random_policy import RandomPolicy
import torch
import gymnasium

import stackelberg_mbrl.envs.simple_mdp
from stackelberg_mbrl.envs.querying_env import LeaderEnv, ModelQueryingEnv, PALLeaderEnv, PolicyQueryingEnv
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
    context_size = len(queries)*real_env.num_actions

    # Pretrain the model conditioned on a policy
    random_policy = RandomPolicy(real_env.num_states, real_env.num_actions, context_size)
    contextualized_real_env = PolicyQueryingEnv(
        env=real_env, 
        policy=random_policy, 
        queries=queries, 
        before_reset=random_policy.randomize)
    
    # TODO: remove sanity check
    # trajectories = sample_trajectories(contextualized_real_env, random_policy, 2, 10000)
    # for t in trajectories:
    #     print("........")
    #     print(t.states)
    #     print(t.actions)
    #     print(t.next_states)
    #     print(t.rewards)

    learn_reward = True # TODO: from config
    rewards = None if learn_reward else None
    contextualized_model = ContextualizedWorldModel( # TODO: properly parameterize from config
        context_size=context_size,
        state_dim=real_env.num_states,
        act_dim=real_env.num_actions,
        rewards=rewards
    )

    print("Pretraining world model")
    pretrain_iterations = 1_000 # TODO: from config
    for iter in range(pretrain_iterations):
        # TODO: can potentially re-use samples from real-env?
        # Does some pretraining of the model oracle under random policies
        
        # generate rollouts on the real environment (using the current policy)
        num_trajectories = 250 # TODO: from config
        max_steps = 10_000 # TODO: from config
        trajectories = sample_trajectories(contextualized_real_env, random_policy, num_trajectories, max_steps)

        observations = np.concatenate(trajectories.states)
        actions = np.concatenate(trajectories.actions)
        rewards = np.concatenate(trajectories.rewards)
        observations_next = np.concatenate(trajectories.next_states)

        # use sgd to match the transitions
        fit_mb_size = 16 # TODO: from config
        fit_epochs = 1 # TODO: from config
        dynamics_loss = contextualized_model.fit_dynamics(observations, actions, observations_next, fit_mb_size, fit_epochs) # use CE-loss (but only on state, not the context!)
        
        if learn_reward:
            rewards_loss = contextualized_model.fit_reward(observations, actions, observations_next, rewards, fit_mb_size, fit_epochs)

        if iter % 25 == 0:
            print(f"Pretraining iteration {iter}")
            print(f"\tDynamics Loss: {dynamics_loss}")

            if learn_reward:
                print(f"\tRewards Loss: {rewards_loss}")


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
    # TODO: actually, does this really give the leader the queries? bc it will only observe its sampled answers :/
    # TODO: contextualized_model should be best response to current policy
    # TODO: where do we get context from here? Need to wrap contextualized_model as environment

    # contextualized_model = (context, s), action -> (context, s_next)
    
    # so we need an env that automatically prepends the context and uses the contextualized models prediction to transition to the next state (but only return the actual state)

    contextualized_leader_env = PALLeaderEnv(
        contextualized_model, 
        initial_state=real_env.initial_state,
        queries=queries, 
        final_state=real_env.final_state,
        max_ep_steps=config.env_config.max_episode_steps
    )

    policy_config: PolicyConfig = config.policy_config
    # TODO: need custom PPO implementation that has callback before doing rollouts (so we can load the oracle weights and finetune it to the current policy)
    policy_ppo = PPO( 
        "MlpPolicy",
        contextualized_leader_env,
        policy_kwargs=policy_config.policy_kwargs,
        tensorboard_log=config.output_dir / config.experiment_name / "tb",
    )

    contextualized_leader_env.set_policy(policy_ppo.policy)
    
    total_training_steps = 1_000_000 # TODO: from config
    policy_ppo.learn(total_training_steps, tb_log_name="Policy", reset_num_timesteps=False, progress_bar=True)


if __name__ == "__main__":
    train_contextualized_PAL(model_rl_config)
