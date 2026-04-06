import gymnasium as gym
from src.bargain.policy.methods import policy_iteration

env = gym.make('FrozenLake8x8-v1')
P = env.unwrapped.P
V, pi = policy_iteration(P)

print("Value Function")
print(V)

s = 0
print(f"Policy at state {s}: {pi(s)}")

