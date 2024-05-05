import gym
from gym import spaces
import numpy as np

from util.tensor_util import extract_one_hot_index_inputs

class SimpleMDPEnv(gym.Env):
    def __init__(self):
        self.num_states = 3 # 0 (A), 1 (B), 2 (C)
        self.state = 0  # Initial state
        self.action_space = spaces.Discrete(2)  # Two possible actions: 0 (X) or 1 (Y)
        self.observation_space = spaces.Discrete(self.num_states)
        self.reward_range = (0, 1)

        # transition matrix (state x action -> state)
        self.transitions = {
            # Target       A    B    C      # Current
            # Action X
            0: np.array([[0.1, 0.6, 0.3],   # A 
                         [0.0, 0.2, 0.8],   # B
                         [0.0, 0.0, 0.0]]), # C
            # Action Y
            1: np.array([[1.0, 0.0, 0.0],   # A 
                         [0.5, 0.5, 0.0],   # B
                         [0.0, 0.0, 0.0]])  # C
        }

        # reward matrix (state x action x state -> r)
        self.rewards = {
            # Target       A    B    C      # Current
            # Action X
            0: np.array([[ 10,  -5, 100],   # A 
                         [  0,  -5, 100],   # B
                         [  0,   0,   0]]), # C
            # Action Y
            1: np.array([[ 10,   0,   0],   # A 
                         [ -1,  -5,   0],   # B
                         [  0,   0,   0]])  # C
        }

    def reset(self):
        self.state = 0
        return self.get_obs()
    
    @extract_one_hot_index_inputs
    def is_done(self, state):
        return state == self.num_states - 1
    
    @extract_one_hot_index_inputs
    def reward(self, state, action, next_state):
        return self.rewards[action][state][next_state]

    @extract_one_hot_index_inputs
    def step(self, action):
        old_state = self.state
        self.state = np.random.choice(self.num_states, p=self.transitions[action][self.state])
        return self.get_obs(), self.reward(old_state, action, self.state), self.is_done(self.state), {}
    
    def get_obs(self):
        observation = np.zeros(3)
        observation[self.state] = 1.0
        return observation

    def render(self):
        print(f"Current state: {self.state}")