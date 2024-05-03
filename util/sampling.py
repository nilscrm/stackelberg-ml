import gym
import numpy as np
from typing import list, Callable

from models.nn_dynamics import WorldModel
from models.random_dynamics import RandomWorldModel
from models.gaussian_mlp import MLP

def sample_model_trajectory(
        env_model: RandomWorldModel,
        policy: MLP,
        queries: list[np.ndarray],
        init_state: np.ndarray,
        reward_function: Callable[[np.ndarray, np.ndarray], float],
        termination_function: Callable[[np.ndarray], bool],
        max_steps: int | None = None):
    """ Sample a trajectory from a world model \Hat{M} """
    # TODO: make this fancier, such that the provided policy can already be conditioned on the model or not (depending on the use-case, e.g. MAL vs PAL)

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

        trajectory.append((state, action, reward, next_state))
        state = next_state

        steps += 1

    return trajectory 


def sample_env_trajectory(
        env: gym.Env,
        env_model: WorldModel,
        policy: MLP,
        queries: list[np.ndarray],
        max_steps: int | None = None):
    """ Sample a trajectory from the real environment M """
    # TODO: make this fancier, such that the provided policy can already be conditioned on the model or not (depending on the use-case, e.g. MAL vs PAL)

    # Query model
    query_answers = []
    for (s, a) in queries:
        query_answers.append(env_model.next_state_distribution(s, a))

    state = env.reset()
    done = False
    steps = 0
    trajectory = []

    while not done and (max_steps is None or steps <= max_steps):
        action, _ = policy.get_action(np.concatenate((state, query_answers)))
        next_state, reward, done, info = env.step(action)

        trajectory.append((state, action, reward, next_state))
        state = next_state

        steps += 1

    return trajectory 

    