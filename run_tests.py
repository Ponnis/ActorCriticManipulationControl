import numpy as np
import gym
import inverse_kinematics_gymified
import inverse_kinematics_gymified.envs.forw_kinm
from inverse_kinematics_gymified.envs.inv_kinm import *
import torch
from rlkit.samplers.rollout_functions import multitask_rollout

pathOfNetworksToTest = [
        "./trained_nets/her_sac_experiments/her-sac-fetch-experiment_2021_05_22_22_22_23_0000--s-0/params.pkl"]

# init env
#env = gym.make('custom_fetch-v0')
#env = gym.make('inverse_kinematics-v0')
#env = gym.make('FetchPickAndPlace-v1')

# init net
data = torch.load(pathOfNetworksToTest[0])
policy = data['evaluation/policy'] # policy is equal to agent in rollout
env = gym.make('inverse_kinematics-with-manip-rewards-no-joint-observations-v0')
env.render()
obs = env.reset()
done = False


def policyClassical(robot, desired_goal):
#    del_thet = invKinmQPSingAvoidE_kI(robot, desired_goal)
    del_thet = invKinm_Jac_T(robot, desired_goal)
#    del_thet = invKinm_PseudoInv(robot, desired_goal)
#    del_thet = invKinm_dampedSquares(robot, desired_goal)
#    del_thet = invKinmQP(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidE_kI(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidE_kM(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidManipMax(robot, desired_goal)

    return del_thet


rewards = []
#while not done:
for experiment in range(1):
    #for i in range(100):
    for i in range(1):
        env.render()
#        action = policyClassical(env.robot, obs['desired_goal'])
#        obs, reward, done, info = env.step(action)

        rez = multitask_rollout(env, policy, max_path_length=100, render=True,
                observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)
        #print(rez['rewards'])
#        rewards.append([reward])
#        print(info)
#print('total reward', np.sum(np.array(rewards))) 
print('total reward', np.sum(np.array(rez['rewards'])))
