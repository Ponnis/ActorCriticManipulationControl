import numpy as np
import pickle
import gym
import inverse_kinematics_gymified
import inverse_kinematics_gymified.envs.forw_kinm
from inverse_kinematics_gymified.envs.inv_kinm import *
import torch
import copy

def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
):
    if full_o_postprocess_func:
        def wrapped_fun(env, agent, o):
            full_o_postprocess_func(env, agent, observation_key, o)
    else:
        wrapped_fun = None

    def obs_processor(o):
        return np.hstack((o[observation_key], o[desired_goal_key]))

    paths = rollout(
        env,
        agent,
        max_path_length=max_path_length,
        render=render,
        render_kwargs=render_kwargs,
        get_action_kwargs=get_action_kwargs,
        preprocess_obs_for_policy_fn=obs_processor,
        full_o_postprocess_func=wrapped_fun,
    )
    if not return_dict_obs:
        paths['observations'] = paths['observations'][observation_key]
    return paths

def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    manip_indexes = []
    smallest_eigenvals = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        manip_indexes.append(inverse_kinematics_gymified.envs.utils.calculateManipulabilityIndex(env.robot))
        smallest_eigenvals.append(inverse_kinematics_gymified.envs.utils.calculateSmallestManipEigenval(env.robot))

        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
        manip_indexes=manip_indexes,
        smallest_eigenvals=smallest_eigenvals
    )


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


results['sac_her'] = {'done': 0, 'rewards': 0, 'final_distance': 0, 'smallest_eigenvalue':0, 'manip_index': 0}
for experiment in range(nExperiments):
    env.reset()
    print('sac_her', experiment)

    rez = multitask_rollout(env, policy, max_path_length=nSteps, render=True,
            observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)
    print(rez['manip_indexes'])
    results['sac_her']['rewards'] += sum(rez['rewards'])
    results['sac_her']['done'] += rez['env_infos'][-1]['is_success']
    # TODO change below so that it's correct
    results['sac_her']['final_distance'] += inverse_kinematics_gymified.envs.utils.goal_distance(obs['achieved_goal'], env.goal)

results['sac_her']['rewards'] = results['sac_her']['rewards'] / nExperiments 
results['sac_her']['done'] = results['sac_her']['done'] / nExperiments 
results['sac_her']['final_distance'] = results['sac_her']['final_distance'] / nExperiments


