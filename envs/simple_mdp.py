from gymnasium import spaces
import numpy as np

from envs.env_util import DiscreteEnv

class SimpleMDPEnv(DiscreteEnv):
    def __init__(self, max_episode_steps: int):
        super().__init__(max_episode_steps)
        self.num_states = 3 # 0 (A), 1 (B), 2 (C)
        self.initial_state = 1  # Initial state
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
            0: np.array([[ .1,  -.05, 1],   # A 
                         [  0,  -.05, 1],   # B
                         [  0,   0,   0]]), # C
            # Action Y
            1: np.array([[ .10,   0,   0],   # A 
                         [ -.01, -.05, 0],   # B
                         [  0,    0,   0]])  # C
        }

        self.step_cnt = 0

        self.reset()

    def reset(self, seed=None):
        self.step_cnt = 0
        self.state = self.initial_state
        return self.get_obs(), {}
    
    def is_done(self, state):
        return state == self.num_states - 1
    
    def reward(self, state, action, next_state):
        return self.rewards[action][state][next_state]

    def step(self, action):
        old_state = self.state
        
        self.state = np.random.choice(self.num_states, p=self.transitions[action][self.state])
        self.step_cnt += 1
        
        truncated = self.step_cnt >= self.max_episode_steps

        return self.get_obs(), self.reward(old_state, action, self.state), self.is_done(self.state), truncated, {}
    
    def get_obs(self):
        return self.state

    def render(self):
        print(f"Current state: {self.state}")