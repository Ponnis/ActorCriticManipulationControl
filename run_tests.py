import numpy as np
import gym
import inverse_kinematics_gymified
import inverse_kinematics_gymified.envs.forw_kinm
from inverse_kinematics_gymified.envs.inv_kinm import *
import torch

pathOfNetworksToTest = [
        "./trained_nets/her_sac_experiments/her-sac-fetch-experiment_2021_05_22_22_22_23_0000--s-0/params.pkl"]

# init env
#env = gym.make('custom_fetch-v0')
#env = gym.make('inverse_kinematics-v0')
#env = gym.make('FetchPickAndPlace-v1')

# init net
data = torch.load(pathOfNetworksToTest[0])
agent = data['evaluation/policy'] # policy is equal to agent in rollout
env = data['evaluation/env']
env.render()
obs = env.reset()
done = False


#raw_obs = []
#raw_next_obs = []
#observations = []
#actions = []
#rewards = []
#terminals = []
#agent_infos = []
#env_infos = []
#next_observations = []
#path_length = 0
#agent.reset()
#o = env.reset()
#while path_length < max_path_length:
#    raw_obs.append(o)
#    a, agent_info = agent.get_action(o)
#
#    next_o, r, d, env_info = env.step(copy.deepcopy(a))
#    observations.append(o)
#    rewards.append(r)
#    terminals.append(d)
#    actions.append(a)
#    next_observations.append(next_o)
#    raw_next_obs.append(next_o)
#    agent_infos.append(agent_info)
#    env_infos.append(env_info)
#    path_length += 1
#    if d:
#        break
#    o = next_o
#actions = np.array(actions)
#if len(actions.shape) == 1:
#    actions = np.expand_dims(actions, 1)
#observations = np.array(observations)
#next_observations = np.array(next_observations)
#if return_dict_obs:
#    observations = raw_obs
#    next_observations = raw_next_obs
#rewards = np.array(rewards)
#if len(rewards.shape) == 1:
#    rewards = rewards.reshape(-1, 1)
#return dict(
#    observations=observations,
#    actions=actions,
#    rewards=rewards,
#    next_observations=next_observations,
#    terminals=np.array(terminals).reshape(-1, 1),
#    agent_infos=agent_infos,
#    env_infos=env_infos,
#    full_observations=raw_obs,
#    full_next_observations=raw_obs,
#)


def policyClassical(robot, desired_goal):
#    del_thet = invKinmQPSingAvoidE_kI(robot, desired_goal)
#    del_thet = invKinm_Jac_T(robot, desired_goal)
#    del_thet = invKinm_PseudoInv(robot, desired_goal)
#    del_thet = invKinm_dampedSquares(robot, desired_goal)
    del_thet = invKinmQP(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidE_kI(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidE_kM(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidManipMax(robot, desired_goal)

    return del_thet



while not done:
    env.render()
    #action = policy(obs['observation'], obs['desired_goal'])
#    action = policy(env.robot, obs['desired_goal'])
    action, info = agent.get_action(obs)
    obs, reward, done, info = env.step(action)
    print(info)

    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = env.compute_reward(
        obs['achieved_goal'], substitute_goal, info)
    print('reward is {}, substitute_reward is {}'.format(
        reward, substitute_reward))

