import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import envs

from tqdm import tqdm

env = gym.make('Bargain-v0'); env.reset()

from gymnasium.utils.env_checker import check_env
try:
    check_env(env.unwrapped)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")

class Agent():

    def __init__(self, env, precision = 10, gamma : float = 0.99):

        self.env = env
        self.precision = precision
        self.gamma = gamma


    def select_action(self, obs, eps : float = 0.1):

        if np.random.uniform() < eps:
            action = np.random.randint(self.env.action_space.n)
        else:
            state = self.get_state_index(obs)
            action = np.argmax(self.Q[state, :])

        return action
    
    def get_state_index(self, obs):

        _low = self.env.observation_space.low
        _high = self.env.observation_space.high
        _range = _high - _low
        """
        The index of a state (obs) is determined by normalizating the range of 
        possible offers and the current offer between 0 and 1 and return the integer 
        that corresponds to position of the current offer in the Q table.
        """
        index = ((obs - _low) / _range) * self.precision

        return int(index[0])
    
    def update(self, obs, action, reward, next_obs):

        state = self.get_state_index(obs)
        next_state = self.get_state_index(next_obs)

        self.N[state, action] += 1 
        self.Q[state, action] += (reward - self.Q[state, action]) / self.N[state, action]
        
    def learn(self, eps_min : float = 0.01, n_episodes : int = 1000):

        self.nA = env.action_space.n
        self.Q = np.zeros((self.precision, self.nA), dtype = np.float64)
        self.N = np.zeros((self.precision, self.nA), dtype = np.int64)

        print(f'> Running EpsGreedy (min = {eps_min}): \n')

        Qe      = np.empty((n_episodes, 
                            self.precision, self.nA), 
                            dtype = np.float64)
        returns = np.empty(n_episodes, dtype = np.float64)
        actions = np.empty(n_episodes, dtype = np.int64)

        for e in tqdm(range(n_episodes)):
            obs, _ = self.env.reset()
            done = False
            episode_return = 0.0
            last_action = 0

            eps = max(eps_min, 1.0 - e / (n_episodes * 0.8))

            while not done:

                action = self.select_action(obs, eps)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if not done:
                    self.update(obs, action, reward, next_obs)

                episode_return += reward
                last_action = action
                obs = next_obs     

            Qe[e] = self.Q
            returns[e] = episode_return
            actions[e] = last_action

        return returns, Qe, actions

agent = Agent(env)
n_episodes = 10000
returns, Qe, actions = agent.learn(n_episodes = n_episodes)

print(Qe)
print(actions)

plt.plot(range(n_episodes), returns)
plt.show()