import gym
from gym import spaces
import numpy as np

class StateMachineEnv(gym.Env):
    def __init__(self):
        self.num_states = 3 # 0 (A), 1 (B), 2 (C)
        self.state = 0  # Initial state
        self.action_space = spaces.Discrete(2)  # Two possible actions: 0 (X) or 1 (Y)
        self.observation_space = spaces.Discrete(self.num_states)
        self.reward_range = (0, 1)

        # transition matrix (action x state -> state)
        self.transitions = {
            # Target       A    B    C      # Current
            # Action X
            0: np.array([[0.3, 0.7, 0.0],   # A 
                         [0.0, 0.2, 0.8],   # B
                         [0.0, 0.0, 0.0]]), # C
            # Action Y
            1: np.array([[1.0, 0.0, 0.0],   # A 
                         [1.0, 0.0, 0.0],   # B
                         [0.0, 0.0, 0.0]])  # C
        }

        # reward matrix (state)
        self.rewards = np.array([5, -10, 100])

    def reset(self):
        self.state = 0
        return self.get_obs()

    def step(self, action):
        action_id = np.argmax(action).item()

        next_state = np.random.choice(self.num_states, p=self.transitions[action_id][self.state])
        reward = self.rewards[next_state]

        # Check if we're in the terminal state
        done = (next_state == self.num_states - 1)

        self.state = next_state
        return self.get_obs(), reward, done, {}
    
    def get_obs(self):
        observation = np.zeros(3)
        observation[self.state] = 1.0
        return observation

    def render(self):
        print(f"Current state: {self.state}")