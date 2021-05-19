import gym
import gym_custom_fetch
#env = gym.make('FetchReach-v1')
env = gym.make('custom_fetch-v0')
env.reset()
import numpy as np
env.render()
for i in range(1000):
    if i % 10 == 0:
        print(i)
    if i < 50:
        print('zero all')
        observation, reward, done, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])) 
    if i > 50 and i < 100:
        print('1st')
        observation, reward, done, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])) 
    if i > 100 and i < 200:
        print('2nd')
        observation, reward, done, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])) 
    if i > 200 and i < 300:
        print('3rd')
        observation, reward, done, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])) 
    if i > 300 and i < 400:
        print('4th')
        observation, reward, done, info = env.step(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])) 
    if i > 400 and i < 500:
        print('5th')
        observation, reward, done, info = env.step(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])) 
    if i > 500 and i < 600:
        print('6th')
        observation, reward, done, info = env.step(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])) 
    if i > 600 and i < 700:
        print('7th')
        observation, reward, done, info = env.step(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])) 


    env.render()

print(f"observation: {observation['observation']}")
print(f"achieved goal: {observation['achieved_goal']}")
print()
print(f"reward: {reward}")
print(done)
