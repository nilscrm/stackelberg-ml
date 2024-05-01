from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------

register(
    id='state-machine',
    entry_point='mjrl.envs:StateMachineEnv',
    max_episode_steps=50,
)

from mjrl.envs.state_machine import StateMachineEnv
