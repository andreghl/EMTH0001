import numpy as np
import gymnasium as gym

from typing import Any

"""
The agent can either be a responder (0) or a proposer (1)
and based on its role has access to different actions.
"""
ACTIONS = {
    0 : {
        0: 'accept',
        1: 'reject'
    },

    1 : {

    }
}

class Bargain(gym.Env):

    def __init__(self, n_agents : int = 2):

        self.action_space = gym.spaces.MultiDiscrete(nvec = 2 * np.ones(shape = n_agents))