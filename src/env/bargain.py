import numpy as np
import gymnasium as gym

from typing import Any

class Bargaining(gym.Env):

    def __init__(self, low : float = 0, high : float = 100):
        super().__init__(self)

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low = min(low, high), high = max(low, high), dtype = np.float64)
        self.state = 0

        self.low = low
        self.high = high
        self.valuation = low

        self.offers : list = []

    def step(self, action : int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"


    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed = seed)

        self.valuation = np.random.uniform(self.low, self.high)

        obs = None
        info = None

        return obs, info

    def get_offer(self):
        return np.random.uniform(self.low, self.high)