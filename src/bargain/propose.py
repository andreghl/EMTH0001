import numpy as np
import gymnasium as gym

from typing import Any

OBSERVATION = {
    0: 'reject',
    1: 'accept'
}

class Proposal(gym.Env):

    def __init__(self, low : float = 0, high : float = 1):

        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Box(
            low = min(low, high),
            high = max(low, high),
            dtype = np.float64
        )

        self.low : float = min(low, high)
        self.high : float = max(low, high)
        self.offers : list = []
        self.steps = 0

    def reset(self,
              *,
              seed : int | None = None,
              options : dict[str, Any] | None = None):
        super().reset(seed = seed, options = options)

        self.valuation = self.np_random.uniform(self.low, 
                                                self.high)
        self.offers = []
        self.steps = 0

        obs = self.np_random.integers(2)

        return obs, {}
    
    def step(self, action : float):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        obs = self.evaluate(action)
        terminated, truncated = False
        self.steps += 1

        if obs == 1:
            reward = self.high - action
            terminated = True
        else:
            reward = - 0.05

        return obs, reward, terminated, truncated, {}

    def evaluate(self, offer : float):
        return (offer >= self.valuation) * 1
