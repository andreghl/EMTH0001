from gymnasium.envs.registration import register
from bargain.respond import Response
from bargain.propose import Proposal

register(
    id = 'Response-v0',
    entry_point = Response,
    max_episode_steps = 1000
    )

register(
    id = 'Proposal-v0',
    entry_point = Proposal,
    max_episode_steps = 1000
    )