from gymnasium import spaces
import numpy as np
from pathlib import Path
from typing import List, Literal

from stackelberg_mbrl.envs.env_util import DiscreteEnv, draw_mdp
from stackelberg_mbrl.util.tensor_util import extract_one_hot_index_inputs

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


def simple_mdp_1(max_episode_steps: int):
    return SimpleMDPEnv(max_episode_steps, transitions, rewards_1, final_state=2)

def simple_mdp_1_variant(max_episode_steps: int):
    return SimpleMDPEnv(max_episode_steps, transitions_variant, rewards_1, final_state=2)

def simple_mdp_2(max_episode_steps: int):
    return SimpleMDPEnv(max_episode_steps, transitions, rewards_2, final_state=2)

def simple_mdp_2_variant(max_episode_steps: int):
    return SimpleMDPEnv(max_episode_steps, transitions_variant, rewards_1, final_state=2)

def ergodic_mdp_1(max_episode_steps: int):
    return SimpleMDPEnv(max_episode_steps, transitions_ergodic_1, rewards_2)


class SimpleMDPEnv(DiscreteEnv):
    def __init__(self, max_episode_steps: int, transition_probs: np.ndarray, rewards: np.ndarray, final_state: int = -1):
        super().__init__(max_episode_steps)
        self.num_states = 3 # 0 (A), 1 (B), 2 (C)
        self.initial_state = 1  # Initial state
        self.action_space = spaces.Discrete(2)  # Two possible actions: 0 (X) or 1 (Y)
        self.observation_space = spaces.Discrete(self.num_states)
        self.reward_range = (-0.05, 1)
        self.final_state = final_state

        # transition matrix (state x action -> state)
        self.transitions = transition_probs

        # reward matrix (state x action x state -> r)
        self.rewards = rewards

        self.step_cnt = 0

        self.reset()

    def reset(self, seed=None):
        self.step_cnt = 0
        self.state = self.initial_state
        return self.get_obs(), {}
    
    def is_done(self, state):
        return state == self.final_state
    
    @extract_one_hot_index_inputs
    def reward(self, state: int, action: int, next_state: int):
        return float(self.rewards[state][action][next_state])

    def step(self, action: int):
        old_state = self.state
        
        self.state = np.random.choice(self.num_states, p=self.transitions[self.state][action])
        self.step_cnt += 1
        
        truncated = self.step_cnt >= self.max_episode_steps

        return self.get_obs(), self.reward(old_state, action, self.state), self.is_done(self.state), truncated, {}
    
    def get_obs(self):
        return self.state

    def render(self):
        print(f"Current state: {self.state}")

    def draw_mdp(self, filepath: Path, format: Literal['png', 'svg'] = 'png'):
        draw_mdp(self.transitions, self.rewards, filepath, format)
