import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import bargain

from bots import Responder
from utils import ma, check

env = gym.make('Response-v0'); env.reset()
check(env)

precision = 100
n_episodes = 10000000

agent = Responder(env, precision)
V, pi, track = agent.q_learning(n_episodes = n_episodes)
size = 1000

plt.plot(ma(track['val'], size), c = 'green', label = 'valuation')
plt.plot(ma(track['reject'], size), c = 'red', label = 'reject')
plt.plot(track['returns'], alpha = 0.1, c = 'blue')
plt.plot(ma(track['returns'], size), c = 'blue', label = 'returns')
plt.legend(loc = 'best')
plt.show()
 
plt.plot(np.argmax(agent.Q, axis = 1))
plt.title("Policy: {0: reject, 1: accept}")
plt.show()