from gymnasium.envs.registration import register
from envs.bargain import Bargain

register(
    id = "Bargain-v0",
    entry_point = Bargain,
    max_episode_steps = 100
    )