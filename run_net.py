import numpy as np
import pickle
import gym
import inverse_kinematics_gymified
import inverse_kinematics_gymified.envs.forw_kinm
from inverse_kinematics_gymified.envs.inv_kinm import *
import torch
from rlkit.samplers.rollout_functions import multitask_rollout

pathOfNetworksToTest = [
        "./trained_nets/her_sac_ik_gymified_with_manip_rewards_final_5_2021_05_24_22_53_24_0000--s-0/params.pkl"]

# init env
#env = gym.make('custom_fetch-v0')
#env = gym.make('inverse_kinematics-v0')
#env = gym.make('FetchPickAndPlace-v1')

# init net
data = torch.load(pathOfNetworksToTest[0])
policy = data['evaluation/policy'] # policy is equal to agent in rollout
#env = gym.make('inverse_kinematics-with-manip-rewards-no-joint-observations-v0')
env = gym.make('inverse_kinematics-with-manip-rewards-v0')
env.render()
obs = env.reset()
done = False




results = {}
nExperiments = 100
nSteps = 50


results['sac_her'] = {'done': 0, 'rewards': 0, 'final_distance': 0}
for experiment in range(nExperiments):
    env.reset()
    print('sac_her', experiment)

    rez = multitask_rollout(env, policy, max_path_length=nSteps, render=True,
            observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)
    #print(rez['rewards'])
    results['sac_her']['rewards'] += sum(rez['rewards'])
    results['sac_her']['done'] += rez['env_infos'][-1]['is_success']
    # TODO change below so that it's correct
    results['sac_her']['final_distance'] += inverse_kinematics_gymified.envs.utils.goal_distance(obs['achieved_goal'], env.goal)

results['sac_her']['rewards'] = results['sac_her']['rewards'] / nExperiments 
results['sac_her']['done'] = results['sac_her']['done'] / nExperiments 
results['sac_her']['final_distance'] = results['sac_her']['final_distance'] / nExperiments


