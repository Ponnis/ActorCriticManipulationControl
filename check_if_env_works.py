import numpy as np
import gym
import inverse_kinematics_gymified

#env = gym.make('custom_fetch-v0')
env = gym.make('inverse_kinematics-v0')
#env = gym.make('FetchPickAndPlace-v1')
env.render()
obs = env.reset()
done = False
print(env.action_space)

def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.

    #return env.action_space.sample()
    return np.ones(6)

for i in range(20000):
    while not done:
        env.render()
        action = policy(obs['observation'], obs['desired_goal'])
        print(action)
        obs, reward, done, info = env.step(action)
    
        # If we want, we can substitute a goal here and re-compute
        # the reward. For instance, we can just pretend that the desired
        # goal was what we achieved all along.
        substitute_goal = obs['achieved_goal'].copy()
        substitute_reward = env.compute_reward(
            obs['achieved_goal'], substitute_goal, info)
        print('reward is {}, substitute_reward is {}'.format(
            reward, substitute_reward))
    
