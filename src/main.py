import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import bargain

from bots import Responder
from utils import ma, check

env = gym.make('Response-v0', sigma = 0.01); env.reset()
check(env)

precision = 100
n_episodes = 1000000
size = 1000

agent = Responder(env, precision)
V, pi, track = agent.qlearn(n_episodes = n_episodes)

plt.plot(track['val'], alpha = 0.2, c = 'green')
plt.plot(ma(track['val'], 100), c = 'green', label = 'valuation')
plt.plot(ma(track['reject'], size), c = 'red', label = 'reject')
plt.plot(track['returns'], alpha = 0.1, c = 'blue')
plt.plot(ma(track['returns'], size), c = 'blue', label = 'returns')
plt.legend(loc = 'best')
plt.show()
 
plt.plot([pi(s) for s in range(precision)])
plt.title("Policy: {0: reject, 1: accept}")
plt.show()

agent.reset()
V, pi, track = agent.eps_greedy(n_episodes = n_episodes)

plt.plot(track['val'], alpha = 0.2, c = 'green')
plt.plot(ma(track['val'], 100), c = 'green', label = 'valuation')
plt.plot(ma(track['reject'], size), c = 'red', label = 'reject')
plt.plot(track['returns'], alpha = 0.1, c = 'blue')
plt.plot(ma(track['returns'], size), c = 'blue', label = 'returns')
plt.legend(loc = 'best')
plt.show()
 
plt.plot([pi(s) for s in range(precision)])
plt.title("Policy: {0: reject, 1: accept}")
plt.show()