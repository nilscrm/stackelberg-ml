from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------

register(
    id='state-machine',
    entry_point='envs:StateMachineEnv',
    max_episode_steps=50,
)

from envs.simple_mdp import StateMachineEnv
