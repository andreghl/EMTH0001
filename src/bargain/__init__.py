from gymnasium.envs.registration import register
from .respond import Response
from .propose import Proposal
from .bargain import Bargain

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