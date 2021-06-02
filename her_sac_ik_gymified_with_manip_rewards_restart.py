import gym
import inverse_kinematics_gymified

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import torch

def experiment(variant):
    # unwrap the TimeLimitEnv wrapper since we manually termiante after 50 steps
    eval_env = gym.make('inverse_kinematics-with-manip-rewards-v0')
#    eval_env = gym.make('inverse_kinematics-v0')
#    eval_env = gym.make('FetchPush-v1').env
#    eval_env = gym.make('FetchReachDense-v1').env
    expl_env = gym.make('inverse_kinematics-with-manip-rewards-v0')
#    expl_env = gym.make('inverse_kinematics-v0')
    #expl_env = gym.make('FetchPush-v1').env
#    expl_env = gym.make('FetchReachDense-v1').env

    data = torch.load('/home/gospodar/chalmers/ADL/her_sac_ik_gymified_with_manip_rewards_final_4_2021_05_24_16_46_22_0000--s-0/params.pkl')

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    qf1 = data['trainer/qf1']
    qf2 = data['trainer/qf2']
    target_qf1 = data['trainer/target_qf1']
    target_qf2 = data['trainer/target_qf1']
    policy = data['exploration/policy']

    eval_policy = data['evaluation/policy']
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algorithm='HER-SAC',
        version='normal',
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=800,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10000,
            max_path_length=50,
        ),
        sac_trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
    )
    #ptu.set_gpu_mode(True)
    setup_logger('her_sac_ik_gymified_with_manip_rewards_final5_cont', variant=variant)
    experiment(variant)
