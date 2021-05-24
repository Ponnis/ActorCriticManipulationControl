import numpy as np
import pickle
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


def policyClassical(robot, desired_goal, alg):

    if alg == 'invKinm_Jac_T':
        del_thet = invKinm_Jac_T(robot, desired_goal)
    if alg == 'invKinm_PseudoInv':
        del_thet = invKinm_PseudoInv(robot, desired_goal)
    if alg == 'invKinm_dampedSquares':
        del_thet = invKinm_dampedSquares(robot, desired_goal)
    if alg == 'invKinmQP':
        del_thet = invKinmQP(robot, desired_goal)
    if alg == 'invKinmQPSingAvoidE_kI':
        del_thet = invKinmQPSingAvoidE_kI(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidE_kM(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidManipMax(robot, desired_goal)

    return del_thet


results = {}
algs = ['invKinm_Jac_T', 'invKinm_PseudoInv', 'invKinm_dampedSquares', 'invKinmQP', 'invKinmQPSingAvoidE_kI']
nExperiments = 100
nSteps = 50



for alg in algs:
    results[alg] = {'done': 0, 'rewards': 0}
    for experiment in range(nExperiments):
        for i in range(nSteps):
        #for i in range(1):
            env.render()
            action = policyClassical(env.robot, obs['desired_goal'], alg)
            obs, reward, done, info = env.step(action)

    #        rez = multitask_rollout(env, policy, max_path_length=100, render=True,
    #                observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)
            #print(rez['rewards'])
            results[alg]['rewards'] += reward
        results[alg]['done'] += info['is_success']

    results[alg]['rewards'] = results[alg]['rewards'] / (nExperiments * nSteps)
    results[alg]['done'] = results[alg]['done'] / (nExperiments * nSteps)


results['sac_her'] = {'done': 0, 'rewards': 0}
for i in range(nExperiments):

    rez = multitask_rollout(env, policy, max_path_length=nSteps, render=False,
            observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)
    #print(rez['rewards'])
    results['sac_her']['rewards'] += sum(rez['rewards'])

    results['sac_her']['done'] += rez['env_infos'][-1]['is_success']

results['sac_her']['rewards'] = results['sac_her']['rewards'] / nExperiments 
results['sac_her']['done'] = results['sac_her']['done'] / nExperiments 

file = file('results_1', 'wb')
pickle.dump(results, file)
file.close()

