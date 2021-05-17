# gym people have this in env's __init__, so i'll do the same then
#from gym.envs.registration import register

#register(id='custom_fetch-v0', entry_point='gym_custom_fetch.envs:CustomFetch',)


# i strait up just copied gym's init, everything here looks reasonable
import distutils.version
import os
import sys
import warnings

from gym import error
from gym.version import VERSION as __version__

from gym.core import Env, GoalEnv, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
from gym.spaces import Space
from gym.envs import make, spec, register
from gym import logger
from gym import vector
from gym import wrappers

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]


# this was here before, now i'm replacing with gym sauce
#from gym_custom_fetch.envs.custom_fetch_env import CustomFetch
from gym.envs.registration import registry, register, make, spec

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
            'reward_type': reward_type,
    }


    print('im about to register custom_fetch{}-v0'.format(suffix))
    register(
        id='custom_fetch{}-v0'.format(suffix),
        entry_point='gym_custom_fetch.envs.robotics:FetchCustomEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )


