from gym.envs.registration import register

register(
    id='inverse_kinematics-v0',
    entry_point='inverse_kinematics_gymified.envs:InverseKinematicsEnv',
)
