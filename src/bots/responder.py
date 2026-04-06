import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

from tqdm import tqdm
from typing import Any

def decay(init_value, min_value, decay_ratio, 
          max_steps, log_start=-2, log_base=10):
    """
    taken from the book
    """
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base = log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values

class Responder():

    def __init__(self, env, precision : int = 10):
        self.env = env
        self.precision = precision

        self.Q = np.zeros((self.precision,
                           self.env.action_space.n), dtype = np.float64)
        
        self.N = np.zeros((self.precision,
                           self.env.action_space.n), dtype = np.int64)
        
    def reset(self, precision : int | None = None):

        if precision:
            self.precision = precision

        self.Q = np.zeros((self.precision,
                           self.env.action_space.n), dtype = np.float64)
        self.N = np.zeros((self.precision,
                           self.env.action_space.n), dtype = np.int64)
        
    def get_state_index(self, obs):

        _low = self.env.observation_space.low
        _high = self.env.observation_space.high
        _range = _high - _low

        """
        The index of a state (obs) is determined by normalizing the range of possible offers and the current offer between 0 and 1 and returning the integer that corresponds to the position of the current offer in the Q table.
        """
        index = (((obs - _low) / _range) * self.precision)[0]
        return min(int(index), self.precision - 1)

    def select_action(self, obs, eps : float = 0.1):

        if self.env.np_random.uniform() < eps:
            action = self.env.action_space.sample()
        else:
            state = self.get_state_index(obs)
            action = np.argmax(self.Q[state])
        return action
    
    def q_learning(self,
            gamma : float = 1.0,
            alpha : tuple[float, float] = (0.5, 0.01, 0.5),
            eps : tuple[float, float] = (1.0, 0.1, 0.9),
            n_episodes : int = 1000):
        
        track : dict[str, Any] = {
            'Q': np.zeros((n_episodes,
                           self.precision,
                           self.env.action_space.n), dtype = np.float64),
            'pi': [],
            'returns': np.zeros(n_episodes, dtype = np.float64),
            'actions': np.zeros((n_episodes,
                                 self.env.action_space.n), dtype = np.int64),
            'reject': np.zeros(n_episodes, dtype = np.float64),
            'iter': np.zeros(n_episodes, dtype = np.int64),
            'val': np.zeros(n_episodes, dtype = np.float64)
            }

        alpha = decay(*alpha, n_episodes)
        eps = decay(*eps, n_episodes)

        for e in tqdm(range(n_episodes), leave = False):
            obs, _ = self.env.reset()
            
            done = False

            while not done:
                action = self.select_action(obs, eps[e])
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                state = self.get_state_index(obs)
                next_state = self.get_state_index(next_obs)

                td_target = reward + gamma * self.Q[next_state].max() * (not done)
                td_error = td_target - self.Q[state][action]
                self.Q[state, action] += alpha[e] * td_error
                track['returns'][e] += reward
                track['iter'][e] += 1
                track['actions'][e, action] += 1

                obs = next_obs

            track['Q'][e] = self.Q.copy()
            track['pi'].append(np.argmax(self.Q, axis = 1))
            track['val'][e] = self.env.unwrapped.valuation
            track['reject'][e] = track['actions'][e, 0] / np.sum(track['actions'][e, :])


        V = np.max(self.Q, axis = 1)
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(self.Q, axis = 1))}[s]

        return V, pi, track