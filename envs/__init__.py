from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------

register(
    id='simple-mdp',
    entry_point='envs:SimpleMDPEnv',
    max_episode_steps=50,
)

from envs.simple_mdp import SimpleMDPEnv
