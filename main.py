import gymnasium as gym
import numpy as np
# import utils.instance

env = gym.make('FrozenLake8x8-v1')
P = env.unwrapped.P

for key in P.keys():
    if key == 0:
        print(key, P[key])

def policy_evaluation(pi, P, gamma = 1.0, theta = 1e-10):
    """
    The policy evaluation algorithm only requires the specific policy 'pi' of interest
    and the MDP 'P' the policy runs on.
    """
    prev_V = np.zeros(len(P))
    while True:
        # The forever loop that stops when the improvement between 'prev_V' and 'V' is smaller
        # than the value of the argument 'theta'.
        V = np.zeros(len(P))
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        """
        Here, the inclusion of '(not done)' ensure the value of any next state when
        landing on a terminal state is always 0 to avoid infinite sums.
        """
        if np.max(np.abs(prev_V - V)) < theta:
            """
            If the state-value function stop changing (enough), we says that they have
            converged and stop the algorithm.
            """
            break
        prev_V = V.copy()
    return V

def policy_improvement(V, P, gamma = 1.0):
    """
    The arguments are the state-value function 'V' of the policy you want to improve,
    the MDP 'P', and optionally the discount factor 'gamma'.
    """
    Q = np.zeros((len(P), len(P[0])), dtype = np.float64)

    for s in range(len(P)):
        for a in range(len(P[s])):
            """
            We loop through the states, actions, and transitions to compute the Q-function.
            """
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    # We obtain a greedy policy by take for each state the action that maximizes
    # the Q-function.
    new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis = 1))}[s]
    return new_pi

def policy_iteration(P, gamma = 1.0, theta = 1e-10):
    # We create a list of random actions and map those to states
    # to create a randomly generated policy.
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s : a for s, a in enumerate(random_actions)}[s]

    while True:
        old_pi = {s : pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)

        if old_pi == {s: pi(s) for s in range(len(P))}:
            break

    return V, pi

V, pi = policy_iteration(P)
print(V)
s = 0
print(pi(s))
print(P[8])
