from gym.envs.registration import register

register(
    id='inverse_kinematics-v0',
    entry_point='inverse_kinematics_gymified.envs:InverseKinematicsEnv',
)


register(
    id='inverse_kinematics-with-manip-rewards-v0',
    entry_point='inverse_kinematics_gymified.envs:InverseKinematicsWithManipRewardsEnv',
)


register(
    id='inverse_kinematics-with-manip-rewards-no-joint-observations-v0',
    entry_point='inverse_kinematics_gymified.envs:InverseKinematicsWithManipRewardsNoJointObservationsEnv',
)
