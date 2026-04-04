import numpy as np
import gymnasium as gym

from typing import Any

ACTIONS = {
    0: 'reject',
    1: 'accept'
}

class Bargain(gym.Env):

    def __init__(self, low : float = 0, high : float = 1):

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low = min(low, high), 
                                                high = max(low, high), 
                                                dtype = np.float64)

        self.low = min(low, high)
        self.high = max(low, high)
        self.valuation = self.low

        self.offers : list = []

    def step(self, action : int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        info = {}

        if action == 0:
            # rejects offer
            observation = self.get_offer()
            reward = 1 if self.valuation > self.offers[-1] else -0.1
            terminated = False
            truncated = False
            
        elif action == 1:
            # accepts offer
            observation = np.array([self.offers[-1]], dtype = np.float64)
            reward = 1 if self.offers[-1] >= self.valuation else -0.1
            terminated = True
            truncated = False
        else:
            observation = self.get_offer()
            reward = -1
            terminated = False
            truncated = False

        return observation, reward, terminated, truncated, info


    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed = seed, options = options)

        self.valuation = self.np_random.uniform(self.low, self.high)
        self.offers = []

        obs = self.get_offer()
        info = {}

        return obs, info

    def get_offer(self):
        offer = self.np_random.uniform(self.low, self.high)
        self.offers.append(offer)
        return np.array([offer], dtype = np.float64)

"""
from gymnasium.utils.env_checker import check_env

# This will catch many common issues
try:
    check_env(env.unwrapped)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
"""