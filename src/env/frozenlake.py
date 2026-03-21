import gymnasium as gym

"""
The goal of this python script is to train an RL agent to navigate the Frozen Lake environment
introduced in Chapter 3 using the techniques proposed in Chapter 4 of the book 'Grokking Deep 
Reinforcement Learning'. 
"""

from env.strategy.greedy import epsilon_greedy

env = gym.make('FrozenLake-v1')
epsilon = 0.01
n_episodes = 10000

name, returns, Qe, actions = epsilon_greedy(env, epsilon, n_episodes)

print(Qe[n_episodes - 1])

print(env.unwrapped.P)