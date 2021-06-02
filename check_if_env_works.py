import numpy as np
import gym
import inverse_kinematics_gymified

env = gym.make('custom_fetch-v0')
#env = gym.make('inverse_kinematics-with-manip-rewards-v0')
#env = gym.make('FetchPickAndPlace-v1')
env.render()
obs = env.reset()
done = False
print(env.action_space)

def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.

    #return env.action_space.sample()
    #return np.array([1,0,0,0,0,0], dtype=np.float32)
    return np.array([-1,0,0,0,0,0], dtype=np.float32)
    #return -1 * np.ones(6)

for i in range(20000):
    while not done:
        env.render()
        action = policy(obs['observation'], obs['desired_goal'])
        print(action)
        obs, reward, done, info = env.step(action)
        print(obs)
    
    
