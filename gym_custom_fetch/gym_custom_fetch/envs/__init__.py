# this was here before, now i'm replacing with gym sauce
#from gym_custom_fetch.envs.custom_fetch_env import CustomFetch
#from gym.envs.registration import registry, register, make, spec
#
#for reward_type in ['sparse', 'dense']:
#    suffix = 'Dense' if reward_type == 'dense' else ''
#    kwargs = {
#            'reward_type': reward_type,
#    }
#    print('im about to register custom_fetch-v0')
#    register(
#        id='custom_fetch-v0',
#        entry_point='gym_custom_fetch.envs.robotics:FetchCustomEnv',
#        kwargs=kwargs,
#        max_episode_steps=50,
#    )
#
#
