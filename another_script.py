import gym
from gym import envs
env = gym.make('FetchReach-v1')
env.reset()
import numpy as np
env.render()
for i in range(1000):
    if i % 10 == 0:
        print(i)
    if i < 100:
        observation, reward, done, info = env.step(np.array([.5,0.5,0.0,1])) # take a random action
    if i > 100 < 200:
        observation, reward, done, info = env.step(np.array([0.0,.5,0.0,0])) # take a random action
    if i > 200 < 300:
        observation, reward, done, info = env.step(np.array([0.0,0.0,0.5,1])) # take a random action
    env.render()

print(f"observation: {observation['observation']}")
print(f"achieved goal: {observation['achieved_goal']}")
print()
print(f"reward: {reward}")
print(done)
