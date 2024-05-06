from gymnasium import spaces
import numpy as np

from envs.env_util import DiscreteEnv
from util.tensor_util import extract_one_hot_index_inputs

def simple_mdp_v0(max_episode_steps: int):
    transitions = [
        #  A    B    C      # <- Target
        # Action X          # Current
        [[0.1, 0.6, 0.3],   # A 
         [0.0, 0.2, 0.8],   # B
         [0.0, 0.0, 1.0]],  # C
        # Action Y
        [[1.0, 0.0, 0.0],   # A 
         [0.5, 0.5, 0.0],   # B
         [0.0, 0.0, 1.0]]   # C
    ]

    return SimpleMDPEnv(max_episode_steps, transitions)

def simple_mdp_v0_variant(max_episode_steps: int):
    transitions = [
        #  A    B    C      # <- Target
        # Action X          # Current
        [[0.1, 0.6, 0.3],   # A 
         [0.0, 1.0, 0.0],   # B
         [0.0, 0.0, 1.0]],  # C
        # Action Y
        [[1.0, 0.0, 0.0],   # A 
         [0.5, 0.5, 0.0],   # B
         [0.0, 0.0, 1.0]]   # C
    ]

    return SimpleMDPEnv(max_episode_steps, transitions)
    


class SimpleMDPEnv(DiscreteEnv):
    def __init__(self, max_episode_steps: int, transition_probs: np.ndarray):
        super().__init__(max_episode_steps)
        self.num_states = 3 # 0 (A), 1 (B), 2 (C)
        self.initial_state = 1  # Initial state
        self.action_space = spaces.Discrete(2)  # Two possible actions: 0 (X) or 1 (Y)
        self.observation_space = spaces.Discrete(self.num_states)
        self.reward_range = (-0.05, 1)

        # transition matrix (action x state -> state)
        self.transitions = transition_probs

        # reward matrix (state x action x state -> r)
        self.rewards = [
            #   A      B      C       # <- Target
            # Action X                # Current
            [[ 1.00, -0.05,  1.00],   # A 
             [ 0.00, -0.05,  1.00],   # B
             [ 0.00,  0.00,  0.00]],  # C
            # Action Y
            [[ 0.75,  0.00, 0.00],   # A 
             [-0.01, -0.05, 0.00],    # B
             [  0.00, 0.00, 0.00]]         # C
        ]

        self.step_cnt = 0

        self.reset()

    def reset(self, seed=None):
        self.step_cnt = 0
        self.state = self.initial_state
        return self.get_obs(), {}
    
    def is_done(self, state):
        return state == self.num_states - 1
    
    @extract_one_hot_index_inputs
    def reward(self, state: int, action: int, next_state: int):
        return float(self.rewards[action][state][next_state])

    def step(self, action: int):
        old_state = self.state
        
        self.state = np.random.choice(self.num_states, p=self.transitions[action][self.state])
        self.step_cnt += 1
        
        truncated = self.step_cnt >= self.max_episode_steps

        return self.get_obs(), self.reward(old_state, action, self.state), self.is_done(self.state), truncated, {}
    
    def get_obs(self):
        return self.state

    def render(self):
        print(f"Current state: {self.state}")