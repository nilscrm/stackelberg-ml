import gym
from gym import spaces
import numpy as np

class StateMachineEnv(gym.Env):
    def __init__(self):
        self.num_states = 3 # 0 (A), 1 (B), 2 (C)
        self.initial_state = 1  # Initial state
        self.action_space = spaces.Discrete(2)  # Two possible actions: 0 (X) or 1 (Y)
        self.observation_space = spaces.Discrete(self.num_states)
        self.reward_range = (0, 1)

        # transition matrix (state x action -> state)
        self.transitions = [
            #  A    B    C      # <- Target
            # Action X          # Current
            [[0.1, 0.6, 0.3],   # A 
             [0.0, 0.2, 0.8],   # B
             [0.0, 0.0, 0.0]],  # C
            # Action Y
            [[1.0, 0.0, 0.0],   # A 
             [0.5, 0.5, 0.0],   # B
             [0.0, 0.0, 0.0]]   # C
        ]

        # reward matrix (state x action x state -> r)
        self.rewards = [
            #   A      B      C       # <- Target
            # Action X                # Current
            [[ 1.00, -0.05,  1.00],   # A 
             [ 0.00, -0.05,  1.00],   # B
             [ 0.00,  0.00,  0.00]],  # C
            # Action Y
            [[ 0.75,  0.00,  0.00],   # A 
             [-0.01, -0.05, 0.00],    # B
             [0.0, 0.0, 0.0]]         # C
        ]

        self.step_cnt = 0

        self.reset()

    def reset(self):
        self.step_cnt = 0
        self.state = self.initial_state
        return self.get_obs()

    def is_done(self, state):
        return state == self.num_states - 1

    def reward(self, state: int, action: int, next_state: int):
        return float(self.rewards[action][state][next_state])
    
    def step(self, action):
        action = np.argmax(action).item()
        old_state = self.state
        
        self.state = np.random.choice(self.num_states, p=self.transitions[action][self.state])
        self.step_cnt += 1

        return self.get_obs(), self.reward(old_state, action, self.state), self.is_done(self.state), {}
    
    def get_obs(self):
        observation = np.zeros(3)
        observation[self.state] = 1.0
        return observation

    def render(self):
        print(f"Current state: {self.state}")