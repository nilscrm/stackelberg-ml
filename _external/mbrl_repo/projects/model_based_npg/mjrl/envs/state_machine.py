import gym
from gym import spaces
import numpy as np

class StateMachineEnv(gym.Env):
    # Simple MDP 2
    def __init__(self):
        self.num_states = 3
        self.initial_state = 1
        self.final_state = 2
        self.action_space = spaces.Discrete(2)  # Two possible actions: 0 (X) or 1 (Y)
        self.observation_space = spaces.Discrete(self.num_states)
        self.reward_range = (-0.05, 1)

        # transition matrix (state x action -> state)
        self.transitions = np.array([
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

        # reward matrix (state x action x state -> r)
        self.rewards = np.array([
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

        self.step_cnt = 0

        self.reset()

    def reset(self):
        self.step_cnt = 0
        self.state = self.initial_state
        return self.get_obs()

    def is_done(self, state):
        return state == self.final_state

    def reward(self, state: int, action: int, next_state: int):
        return float(self.rewards[state][action][next_state])
    
    def step(self, action):
        action = np.argmax(action).item()
        old_state = self.state
        
        self.state = np.random.choice(self.num_states, p=self.transitions[self.state][action])
        self.step_cnt += 1

        return self.get_obs(), self.reward(old_state, action, self.state), self.is_done(self.state), {}
    
    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        acs = paths["actions"]
        num_traj, horizon, obs_dim = obs.shape
        
        state_indices = np.argmax(obs, axis=-1)
        action_indices = np.argmax(acs, axis=-1)

        rewards = np.zeros((num_traj, horizon))
        for i in range(num_traj):
            prev_state_idx = self.initial_state
            for j in range(horizon):
                next_state_idx = state_indices[i,j]
                rewards[i,j] = self.reward(prev_state_idx, action_indices[i,j], next_state_idx)
                prev_state_idx = next_state_idx

        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
        return paths

    def get_obs(self):
        observation = np.zeros(3)
        observation[self.state] = 1.0
        return observation

    def render(self):
        print(f"Current state: {self.state}")