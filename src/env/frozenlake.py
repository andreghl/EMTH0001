import numpy as np
import gymnasium as gym

"""
The goal of this python script is to train an RL agent to navigate the Frozen Lake environment
introduced in Chapter 3 using the techniques proposed in Chapter 4 of the book 'Grokking Deep 
Reinforcement Learning'. 
"""

env = gym.make('FrozenLake8x8-v1')