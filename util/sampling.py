import gym
import numpy as np
from typing import list, Callable

from models.nn_dynamics import WorldModel
from models.gaussian_mlp import MLP

def sample_model_trajectory(
        env_model: WorldModel,
        policy: MLP,
        queries: list[np.ndarray],
        init_state: np.ndarray,
        reward_function: Callable[[np.ndarray, np.ndarray], float],
        termination_function: Callable[[np.ndarray], bool],
        max_steps: int | None = None):
    # Query model
    query_answers = []
    for (s, a) in queries:
        query_answers.append(env_model.next_state_distribution(s, a))

    state = init_state
    done = False
    steps = 0
    trajectory = []

    while not done and (max_steps is None or steps <= max_steps):
        action, _ = policy.get_action(np.concatenate((state, query_answers)))
        reward = reward_function(state, action)
        next_state = env_model.sample_next_state(state, action)
        done = termination_function(next_state)

        trajectory.append((query_answers, state, action, reward, next_state))
        state = next_state

        steps += 1

    return trajectory 


# TODO: Sample trajectories from real environment
def sample_env_trajectory(
        env: gym.Env,
        env_model: WorldModel,
        policy: MLP,
        queries: list[np.ndarray],
        max_steps: int | None = None):
    raise NotImplementedError