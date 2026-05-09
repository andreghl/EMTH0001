import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

from tqdm import tqdm
from typing import Any

from .tools import decay

class Proposer():

    def __init__(self, env : gym.Env, precision : int = 10):
        self.env = env
        self.precision = precision

        self.Q = np.zeros((self.env.observation_space.n,
                           self.precision), dtype = np.float64)
        
        self.N = np.zeros((self.env.observation_space.n,
                           self.precision), dtype = np.int64)
        
    def reset(self, precision : int | None = None):
        if precision:
            self.precision = precision

        self.Q = np.zeros((self.env.observation_space.n,
                           self.precision), dtype = np.float64)
        
        self.N = np.zeros((self.env.observation_space.n,
                           self.precision), dtype = np.int64)
        
    def get_action_index(self, action):

        _low = self.env.action_space.low
        _high = self.env.action_space.high
        _range = _high - _low

        """
        The index of a state (obs) is determined by normalizing the range of possible offers and the current offer between 0 and 1 and returning the integer that corresponds to the position of the current offer in the Q table.
        """
        index = (((action - _low) / _range) * self.precision)[0]
        return min(int(index), self.precision - 1)
    
    def select_action(self, state, eps : float = 0.1):

        if self.env.np_random.uniform() < eps:
            action = self.env.action_space.sample()
        else:
            value = np.argmax(self.Q[state]).astype(float) / self.precision
            action = np.array([value], dtype = np.float64)
        return action

    def qlearn(self,
            gamma : float = 1.0,
            alpha : tuple[float, float] = (0.5, 0.01, 0.5),
            eps : tuple[float, float] = (1.0, 0.1, 0.9),
            n_episodes : int = 1000):
        
        track : dict[str, Any] = {
            'Q': np.zeros((n_episodes,
                           self.env.observation_space.n,
                           self.precision), dtype = np.float64),
            'pi': [],
            'returns': np.zeros(n_episodes, dtype = np.float64),
            'actions': np.zeros((n_episodes,
                                 self.precision), dtype = np.int64),
            'proposal': np.zeros(n_episodes, dtype = np.float64),
            'iter': np.zeros(n_episodes, dtype = np.int64),
            'val': np.zeros(n_episodes, dtype = np.float64)
            }

        alpha = decay(*alpha, n_episodes)
        eps = decay(*eps, n_episodes)

        for e in tqdm(range(n_episodes), leave = False):
            state, _ = self.env.reset()
            done = False

            while not done:
                action = self.select_action(state, eps[e])

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                action = self.get_action_index(action)

                td_target = reward + gamma * self.Q[next_state].max() * (not done)
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += alpha[e] * td_error
                track['returns'][e] += reward
                track['iter'][e] += 1
                track['actions'][e, action] += 1

                state = next_state

            track['Q'][e] = self.Q.copy()
            track['pi'].append(np.argmax(self.Q, axis = 1))
            track['val'][e] = self.env.unwrapped.valuation


        V = np.max(self.Q, axis = 1)
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(self.Q, axis = 1))}[s]

        return V, pi, track
    
    def eps_greedy(self, eps_min : float = 0.01, n_episodes : int = 1000):

        track : dict[str, Any] = {
            'Q': np.zeros((n_episodes,
                           self.env.observation_space.n,
                           self.precision), dtype = np.float64),
            'pi': [],
            'returns': np.zeros(n_episodes, dtype = np.float64),
            'actions': np.zeros((n_episodes,
                                 self.precision), dtype = np.int64),
            'proposal': np.zeros(n_episodes, dtype = np.float64),
            'iter': np.zeros(n_episodes, dtype = np.int64),
            'val': np.zeros(n_episodes, dtype = np.float64)
            }
        
        for e in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            done = False

            eps = max(eps_min, 1.0 - e / (n_episodes * 0.8))

            while not done:

                action = self.select_action(state, eps)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                action = self.get_action_index(action)


                self.N[state, action] += 1 
                self.Q[state, action] += (reward - self.Q[state, action]) / self.N[state, action]
                track['returns'][e] += reward
                track['iter'][e] += 1
                track['actions'][e, action] += 1
                state = next_state

            track['Q'][e] = self.Q.copy()
            track['pi'].append(np.argmax(self.Q, axis = 1))
            track['val'][e] = self.env.unwrapped.valuation

        V = np.max(self.Q, axis = 1)
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(self.Q, axis = 1))}[s]

        return V, pi, track