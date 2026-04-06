import gymnasium as gym
import numpy as np

from tqdm import tqdm

# Import the MABs from magni84/gym_bandits
import gym_bandits

# Compare to behavior with full knowledge of the MDP
from policy.methods import policy_iteration

"""
The goal of this python script is to train an RL agent to navigate a Multi-armed bandit env
introduced in Chapter 4 using the techniques proposed in Chapter 4 of the book 'Grokking Deep 
Reinforcement Learning'. 
"""

env = gym.make("MultiarmedBandits-v0", nr_arms = 2); env.reset()

# A Simple strategy for a Two-armed bandit called 'epsilon greedy exploration'
def epsilon_greedy(env, epsilon = 0.01, n_episodes = 1000):
    Q = np.zeros((env.action_space.n), dtype = np.float64)
    N = np.zeros((env.action_space.n), dtype = np.int64)

    Qe = np.empty((n_episodes, env.action_space.n), dtype=np.float64)
    returns = np.empty(n_episodes, dtype = np.float64)
    actions = np.empty(n_episodes, dtype = np.int64)
    name = 'Epsilon-Greedy {}'.format(epsilon)
    for e in tqdm(range(n_episodes),
                  desc='Episodes for: ' + name,
                  leave=False):
        if np.random.uniform() > epsilon:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))

        _, reward, _, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action])/N[action]

        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
    return name, returns, Qe, actions

def optimistic_initialization(env,
                              optimistic_estimate=1.0,
                              initial_count=100,
                              n_episodes=1000):
    Q = np.full((env.action_space.n), optimistic_estimate, dtype=np.float64)
    N = np.full((env.action_space.n), initial_count, dtype=np.int64)

    Qe = np.empty((n_episodes, env.action_space.n), dtype=np.float64)
    returns = np.empty(n_episodes, dtype=np.float64)
    actions = np.empty(n_episodes, dtype=np.int64)
    name = 'Optimistic {}, {}'.format(optimistic_estimate,
                                      initial_count)
    for e in tqdm(range(n_episodes),
                  desc='Episodes for: ' + name,
                  leave=False):
        action = np.argmax(Q)

        _, reward, _, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action])/N[action]

        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
    return name, returns, Qe, actions

epsilon = 0.01
n_episodes = 100000
name, returns, Qe, actions = epsilon_greedy(env, epsilon, n_episodes)
print(np.sum(returns))
print(np.sum(actions))
name, returns, Qe, actions = optimistic_initialization(env, n_episodes = n_episodes)
print(np.sum(returns))
print(np.sum(actions))


for i in range(5):
    print(f'{0}: {env.step(0)}')
    print(f'{1}: {env.step(1)}')