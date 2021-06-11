import numpy as np
import pickle
import gym
import inverse_kinematics_gymified
import inverse_kinematics_gymified.envs.forw_kinm
from inverse_kinematics_gymified.envs.inv_kinm import *
import torch
import copy

####################################################################
# adjusted rlkit commands 
####################################################################

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
    singularity = 0
    path_length = 0
    agent.reset()
    o = env.reset_test()
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
        rezz = inverse_kinematics_gymified.envs.utils.calculatePerformanceMetrics(env.robot)
        manip_indexes.append(rezz['manip_index'])
        smallest_eigenvals.append(rezz['smallest_eigenval'])
        singularity = rezz['singularity']

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
        smallest_eigenvals=smallest_eigenvals,
        singularity=singularity
    )

####################################################################
# load networks
####################################################################

pathOfNetworksToTest = [
        "./trained_nets/her_sac_ik_gymified_with_manip_rewards_final_5_2021_05_24_22_53_24_0000--s-0/params.pkl",
        "./trained_nets/her-sac-no-manip-rewards_2021_05_25_10_55_18_0000--s-0/params.pkl",
        "./trained_nets/big_better_rewards/params.pkl",
        "./trained_nets/her-sac-stronger-convergence-rewards-neg-plus-inverse_2021_06_02_14_36_52_0000--s-0/params.pkl",
        "./trained_nets/her-sac-stronger-convergence-rewards-neg-plus-inverse-CLAMP_2021_06_07_19_35_09_0000--s-0/params.pkl"]
# init env
#env = gym.make('custom_fetch-v0')
#env = gym.make('inverse_kinematics-v0')
#env = gym.make('FetchPickAndPlace-v1')

# init net
data_manip_rewards = torch.load(pathOfNetworksToTest[0])
data_no_manip_rewards = torch.load(pathOfNetworksToTest[1])
data_big_net_manip_rewards = torch.load(pathOfNetworksToTest[2])
data_pro_convergence_rewards = torch.load(pathOfNetworksToTest[3])
#data_clamp_no_manip_rewards = torch.load(pathOfNetworksToTest[4])

policy_trained_on_manip_rewards = data_manip_rewards['evaluation/policy'] # policy is equal to agent in rollout
policy_no_manip_rewards = data_no_manip_rewards['evaluation/policy'] # policy is equal to agent in rollout
policy_manip_rewards_big_net = data_big_net_manip_rewards['evaluation/policy'] # policy is equal to agent in rollout
policy_pro_convergence_rewards = data_pro_convergence_rewards['evaluation/policy']
#policy_clamp_no_manip_rewards = data_clamp_no_manip_rewards['evaluation/policy']
#env = gym.make('inverse_kinematics-with-manip-rewards-no-joint-observations-v0')
#env = gym.make('inverse_kinematics-with-manip-rewards-v0')
env = gym.make('inverse_kinematics-v0')
#env.render()
obs = env.reset()
done = False


####################################################################
# define classical policies
####################################################################


def policyClassical(robot, desired_goal, alg):

    if alg == 'transpose':
        del_thet = invKinm_Jac_T(robot, desired_goal)
    if alg == 'pseudoinverse':
        del_thet = invKinm_PseudoInv(robot, desired_goal)
    if alg == 'damped squares':
        del_thet = invKinm_dampedSquares(robot, desired_goal)
    if alg == 'invKinmQP':
        del_thet = invKinmQP(robot, desired_goal)
    if alg == 'advanced QP':
        del_thet = invKinmQPSingAvoidE_kI(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidE_kM(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidManipMax(robot, desired_goal)

    return del_thet


####################################################################
# initialize result data structures
####################################################################

results = {}
#algs = ['invKinm_Jac_T', 'invKinm_PseudoInv', 'invKinm_dampedSquares', 'invKinmQP', 'invKinmQPSingAvoidE_kI']
algs = ['transpose', 'pseudoinverse', 'damped squares', 'advanced QP']
#nExperiments = 220
nExperiments = 250
#nExperiments = 2
nSteps = 50

results['meta'] = {'nExperiments': nExperiments, 'nSteps': nSteps}
#-------------------------------------------------------------------
# note on how we'll calculate the standard deviations for manip indexed and smallest eigvals:
# we'll simply put all of the in a single list, essentially pretending that we have
# generated one huge trajectory instead of nExperiment individual ones.
# this the simplest solution which isn't blatantly wrong.
#-------------------------------------------------------------------


####################################################################
# run classical algorithms
####################################################################

if 0 == 0:

    for alg in algs:
        results[alg] = {'successes': 0, 'returns': [], 'final_distances': [], 'smallest_eigenvals': [],
                        'manip_indexes': [], 'singularities' : 0}
        for experiment in range(nExperiments):
            results[alg]['returns'].append(0)
            env.reset_test()
            print(alg, experiment)
            for i in range(nSteps):
                action = policyClassical(env.robot, obs['desired_goal'], alg)
                obs, reward, done, info = env.step(action)

                # save everything relevant
                rezz = inverse_kinematics_gymified.envs.utils.calculatePerformanceMetrics(env.robot)
                results[alg]['returns'][-1] += reward
                results[alg]['smallest_eigenvals'].append(rezz['smallest_eigenval'])
                results[alg]['manip_indexes'].append(rezz['manip_index'])
            results[alg]['successes'] += info['is_success']
            results[alg]['final_distances'].append(inverse_kinematics_gymified.envs.utils.goal_distance(obs['achieved_goal'], env.goal))
            results[alg]['singularities'] += rezz['singularity']
            if rezz['singularity'] == 1:
                env.reset()


    ####################################################################
    # run sac her clamp edition
    ####################################################################

#    results['RL_2'] = {'successes': 0, 'returns': [], 'final_distances': [], 'smallest_eigenvals': [],
#                        'manip_indexes': [], 'singularities' : 0}
#
#    for experiment in range(nExperiments):
#        env.reset_test()
#        print('RL_2', experiment)
#
#        rez = multitask_rollout(env, policy_clamp_no_manip_rewards, max_path_length=nSteps, 
#                render=False,
#                observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)
#        #print(rez['rewards'])
#        results['RL_2']['returns'].append(sum(rez['rewards']))
#        results['RL_2']['successes'] += rez['env_infos'][-1]['is_success']
#        results['RL_2']['final_distances'].append(
#                                                inverse_kinematics_gymified.envs.utils.goal_distance(
#                                                rez['observations'][-1]['achieved_goal'], 
#                                                rez['observations'][-1]['desired_goal']))
#        results['RL_2']['smallest_eigenvals'].append(rez['smallest_eigenvals'])
#        results['RL_2']['manip_indexes'].append(rez['manip_indexes'])
#        results['RL_2']['singularities'] += rez['singularity']
#
#        if rezz['singularity'] == 1:
#            env.reset()


    ####################################################################
    # run sac her with manip rewards
    ####################################################################

    results['RL_2'] = {'successes': 0, 'returns': [], 'final_distances': [], 'smallest_eigenvals': [],
                        'manip_indexes': [], 'singularities' : 0}

    for experiment in range(nExperiments):
        env.reset_test()
        print('RL_2', experiment)

        rez = multitask_rollout(env, policy_trained_on_manip_rewards, max_path_length=nSteps, 
                render=False,
                observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)
        #print(rez['rewards'])
        results['RL_2']['returns'].append(sum(rez['rewards']))
        results['RL_2']['successes'] += rez['env_infos'][-1]['is_success']
        results['RL_2']['final_distances'].append(
                                                inverse_kinematics_gymified.envs.utils.goal_distance(
                                                rez['observations'][-1]['achieved_goal'], 
                                                rez['observations'][-1]['desired_goal']))
        results['RL_2']['smallest_eigenvals'].append(rez['smallest_eigenvals'])
        results['RL_2']['manip_indexes'].append(rez['manip_indexes'])
        results['RL_2']['singularities'] += rez['singularity']

        if rezz['singularity'] == 1:
            env.reset()



   ####################################################################
   # run sac her with distance only rewards
   ####################################################################
    results['RL_1'] = {'successes': 0, 'returns': [], 'final_distances': [], 'smallest_eigenvals': [],
                        'manip_indexes': [], 'singularities' : 0}

    for experiment in range(nExperiments):
        env.reset_test()
        print('RL_1', experiment)

        rez = multitask_rollout(env, policy_no_manip_rewards, max_path_length=nSteps, 
                render=False,
                observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)
        #print(rez['rewards'])
        results['RL_1']['returns'].append(sum(rez['rewards']))
        results['RL_1']['successes'] += rez['env_infos'][-1]['is_success']
        results['RL_1']['final_distances'].append(
                                                inverse_kinematics_gymified.envs.utils.goal_distance(
                                                rez['observations'][-1]['achieved_goal'], 
                                                rez['observations'][-1]['desired_goal']))
        results['RL_1']['smallest_eigenvals'].append(rez['smallest_eigenvals'])
        results['RL_1']['manip_indexes'].append(rez['manip_indexes'])
        results['RL_1']['singularities'] += rez['singularity']
        if rezz['singularity'] == 1:
            env.reset()




####################################################################
# run sac her with distance only rewards and big net
####################################################################
results['RL_2_big_net'] = {'successes': 0, 'returns': [], 'final_distances': [], 'smallest_eigenvals': [],
                    'manip_indexes': [], 'singularities' : 0}

for experiment in range(nExperiments):
    env.reset_test()
    print('RL_2_big_net', experiment)

    rez = multitask_rollout(env, policy_manip_rewards_big_net, max_path_length=nSteps, 
            render=False,
            observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)
    #print(rez['rewards'])
    results['RL_2_big_net']['returns'].append(sum(rez['rewards']))
    results['RL_2_big_net']['successes'] += rez['env_infos'][-1]['is_success']
    results['RL_2_big_net']['final_distances'].append(
                                            inverse_kinematics_gymified.envs.utils.goal_distance(
                                            rez['observations'][-1]['achieved_goal'], 
                                            rez['observations'][-1]['desired_goal']))
    results['RL_2_big_net']['smallest_eigenvals'].append(rez['smallest_eigenvals'])
    results['RL_2_big_net']['manip_indexes'].append(rez['manip_indexes'])
    results['RL_2_big_net']['singularities'] += rez['singularity']
    if rez['singularity'] == 1:
        env.reset()



####################################################################
# run sac her with pro convergence rewards
####################################################################
results['RL_3'] = {'successes': 0, 'returns': [], 'final_distances': [], 'smallest_eigenvals': [],
                    'manip_indexes': [], 'singularities' : 0}

for experiment in range(nExperiments):
    env.reset_test()
    print('RL_3', experiment)

    rez = multitask_rollout(env, policy_pro_convergence_rewards, max_path_length=nSteps, 
            render=False,
            observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)
    #print(rez['rewards'])
    results['RL_3']['returns'].append(sum(rez['rewards']))
    results['RL_3']['successes'] += rez['env_infos'][-1]['is_success']
    results['RL_3']['final_distances'].append(
                                            inverse_kinematics_gymified.envs.utils.goal_distance(
                                            rez['observations'][-1]['achieved_goal'], 
                                            rez['observations'][-1]['desired_goal']))
    results['RL_3']['smallest_eigenvals'].append(rez['smallest_eigenvals'])
    results['RL_3']['manip_indexes'].append(rez['manip_indexes'])
    results['RL_3']['singularities'] += rez['singularity']
    if rez['singularity'] == 1:
        env.reset()

####################################################################
# save results to a file
####################################################################

file = open('results_NO_clamp_FINAL_only_neg_dist_rewards', 'wb')
pickle.dump(results, file)
file.close()

