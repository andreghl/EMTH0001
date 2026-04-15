import numpy as np
import gymnasium as gym

from typing import Any

ACTIONS = {
    0: 'reject',
    1: 'accept'
}

class Response(gym.Env):

    def __init__(self, low : float = 0, high : float = 1, sigma : float = 1.0):

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low = min(low, high), 
            high = max(low, high), 
            dtype = np.float64
            )

        self.low = min(low, high)
        self.high = max(low, high)
        self.sigma = sigma
        self.offers : list = []
        self.steps = 0

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed = seed, options = options)

        v = self.np_random.normal((self.high + self.low) / 2, self.sigma * (self.high - self.low))
        self.valuation = np.clip(v, self.low, self.high)
        self.offers = []
        self.steps = 0

        obs = self.get_offer()
        info = {}

        return obs, info

    def step(self, action : int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        self.steps += 1

        if action == 0:
            # rejects offer
            observation = self.get_offer()
            reward = - 0.01
            terminated = False
            truncated = False
            
        elif action == 1:
            # accepts offer
            observation = np.array([self.offers[-1]], dtype = np.float64)
            reward = self.offers[-1] - self.valuation
            terminated = True
            truncated = False
        else:
            # in case it choose an action that doesn't exist
            # looks like it does not choose it anymore.
            observation = self.get_offer()
            reward = - 100000000
            terminated = False
            truncated = False

        return observation, reward, terminated, truncated, {}

    def get_offer(self):
        offer = self.np_random.normal(self.valuation, self.high - self.low)
        offer = np.clip(offer, self.low, self.high)
        self.offers.append(offer)
        return np.array([offer], dtype = np.float64)