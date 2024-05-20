import numpy as np

from stackelberg_mbrl.envs.env_util import MatrixMDP

rewards_1 = np.array([
    #  A    B    C    <- New State
    # Old State A
    [[ 1.00, -0.05,  1.00],    # Action X
     [ 0.75,  0.00, 0.00]],   # Action Y
    # Old State B
    [[ 0.00, -0.05,  1.00],    # Action X
     [-0.01, -0.05, 0.00]],   # Action Y
    # Old State C
    [[ 0.00,  0.00,  0.00],    # Action X
     [ 0.00,  0.00, 0.00]],   # Action Y
])

rewards_2 = np.array([
    #  A    B    C    <- New State
    # Old State A
    [[ 1.00, -0.05,  1.00],    # Action X
     [ 0.75,  0.00, 0.00]],   # Action Y
    # Old State B
    [[ 0.00, -0.05,  1.00],    # Action X
     [  0.5, -0.05, 0.00]],   # Action Y
    # Old State C
    [[ 0.00,  0.00,  0.00],    # Action X
     [ 0.00,  0.00, 0.00]],   # Action Y
])

transitions = np.array([
    #  A    B    C    <- New State
    # Old State A
    [[0.1, 0.6, 0.3],    # Action X
     [1.0, 0.0, 0.0]],   # Action Y
    # Old State B
    [[0.0, 0.2, 0.8],    # Action X
     [0.5, 0.5, 0.0]],   # Action Y
    # Old State C
    [[0.0, 0.0, 1.0],    # Action X
     [0.0, 0.0, 1.0]],   # Action Y
])

# TODO: chose one that makes more sense (different best policy from our true env)
transitions_variant = np.array([
    #  A    B    C    <- New State
    # Old State A
    [[0.1, 0.6, 0.3],    # Action X
     [1.0, 0.0, 0.0]],   # Action Y
    # Old State B
    [[0.0, 1.0, 0.0],    # Action X
     [0.5, 0.5, 0.0]],   # Action Y
    # Old State C
    [[0.0, 0.0, 1.0],    # Action X
     [0.0, 0.0, 1.0]],   # Action Y
])

transitions_ergodic_1 = np.array([
    #  A    B    C    <- New State
    # Old State A
    [[0.1, 0.6, 0.3],    # Action X
     [1.0, 0.0, 0.0]],   # Action Y
    # Old State B
    [[0.0, 0.2, 0.8],    # Action X
     [0.5, 0.5, 0.0]],   # Action Y
    # Old State C
    [[0.0, 0.1, 0.9],    # Action X
     [0.1, 0.0, 0.9]],   # Action Y
])


def simple_mdp_1(max_episode_steps: int) -> MatrixMDP:
    return MatrixMDP(max_episode_steps, transitions, rewards_1, initial_state=1, final_state=2)

def simple_mdp_1_variant(max_episode_steps: int) -> MatrixMDP:
    return MatrixMDP(max_episode_steps, transitions_variant, rewards_1, initial_state=1, final_state=2)

def simple_mdp_2(max_episode_steps: int) -> MatrixMDP:
    return MatrixMDP(max_episode_steps, transitions, rewards_2, initial_state=1, final_state=2)

def simple_mdp_2_variant(max_episode_steps: int) -> MatrixMDP:
    return MatrixMDP(max_episode_steps, transitions_variant, rewards_1, initial_state=1, final_state=2)

def ergodic_mdp_1(max_episode_steps: int) -> MatrixMDP:
    return MatrixMDP(max_episode_steps, transitions_ergodic_1, rewards_2, initial_state=1)
