import numpy as np
from tqdm import tqdm

def epsilon_greedy(env, epsilon_min = 0.01, n_episodes = 1000):
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    N = np.zeros((n_states, n_actions), dtype=np.int64)

    Qe      = np.empty((n_episodes, n_states, n_actions), dtype=np.float64)
    returns = np.empty(n_episodes, dtype=np.float64)
    actions = np.empty(n_episodes, dtype=np.int64)
    name    = f'Epsilon-Greedy (min ε={epsilon_min})'

    for e in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        last_action = 0

        epsilon = max(epsilon_min, 1.0 - e / (n_episodes * 0.8))

        while not done:
            if np.random.uniform() > epsilon:
                action = np.argmax(Q[obs])
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            N[obs, action] += 1
            Q[obs, action] += (reward - Q[obs, action]) / N[obs, action]

            episode_return += reward
            last_action = action
            obs = next_obs
            done = terminated or truncated

        Qe[e]      = Q
        returns[e] = episode_return
        actions[e] = last_action

    return name, returns, Qe, actions