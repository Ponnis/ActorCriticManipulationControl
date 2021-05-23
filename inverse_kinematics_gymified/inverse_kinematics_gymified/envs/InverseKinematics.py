# gym related imports
import gym
from gym.spaces import Box
from gym import error, spaces, utils
from gym.utils import seeding

# kinematics related imports
from inverse_kinematics_gymified.envs.forw_kinm import *
from inverse_kinematics_gymified.envs.inv_kinm import *
from inverse_kinematics_gymified.envs.follow_curve import *
from inverse_kinematics_gymified.envs.utils import *
from inverse_kinematics_gymified.envs.drawing import *

# general imports
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


"""
IMPORTANT: MAKE EVERYTHING FLOAT32 because you don't need more
and that will be much faster
list of methods which need to be implemented:



list of variables which need to exist and be correct:
    1. observation_space 
    2. action_space
    here you set the number of joints and their limit, ex.:
    deepworlds panda does:
    self.observation_space = Box(low=np.array([-np.inf, -np.inf, -np.inf, -2.8972, -1.7628, -2.8972, -3.0718, -2.8972, -0.0175, -2.8972]),
    high=np.array([np.inf,  np.inf,  np.inf, 2.8972,  1.7628,  2.8972, -0.0698,  2.8972,  3.7525,  2.8972]), dtype=np.float64)
    self.action_space = Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float64)



"""


class InverseKinematicsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

# set damping (i.e. dt i.e. precision let's be real)
    def __init__(self, model_path=None, initial_qpos=None, n_actions=None, n_substeps=None ):
        print('env created')
        # TODO write a convenience dh_parameter loading function
        self.robot = Robot_raw(robot_name="no_sim")
        self.damping = 5
        self.error_vec = None
        # number of timesteps allowed
        self.max_tries = 300
        # maximum number of episodes --> TODO change to appripriate amount
        self.total_number_of_points = 150
        # keep track of number of episodes completed
        self.n_of_points_done = 0
        # keep track of the timesteps (to be reset after every episode)
        self.n_of_tries_for_point = 0

        # TODO init goal
        self.goal = np.random.random(3) * 0.7
        # needed for easy initialization of the observation space
        obs = self._get_obs()

        # select an inteligent place for the file
        # idk what this relative path does
        # also make sure to not override stuff
#        self.measurements_file = open("./data/measurementsXYZ", "w")
        observation_space_low = np.array([-np.inf] * (3 + self.robot.ndof), dtype=np.float32)
        observation_space_high = np.array([np.inf] * (3 + self.robot.ndof), dtype=np.float32)


#        self.observation_space = Box(low=observation_space_low, \
#                                    high=observation_space_high, dtype=np.float64)
        self.observation_space = spaces.Dict(dict(                                                
            desired_goal=spaces.Box(-np.inf, np.inf, 
                                    shape=obs['achieved_goal'].shape, dtype='float32'),           
            achieved_goal=spaces.Box(-np.inf, np.inf, 
                                    shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(low=observation_space_low, \
                                    high=observation_space_high, 
                                    shape=obs['observation'].shape,
                                    dtype='float32'),
        ))

        self.action_space = Box(low=np.array([-1.0] * self.robot.ndof, dtype=np.float32),  \
                                high=np.array([1.0] * self.robot.ndof, dtype=np.float32), dtype=np.float32)

        # TODO enable setting the other one with greater ease
        self.reward_type = 'dense'
        self.episode_score = 0
        self.drawingInited = False


    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        if self.reward_type == 'sparse':
            if error_test(self.robot.p_e, self.goal):
                return np.float32(-1.0)
            else:
                return np.float32(1.0)
        if self.reward_type == 'dense':
            distance = goal_distance(achieved_goal, goal)
            reward = -1 * distance
            # add a lil extra if it's close
            if distance < 0.01:
                reward = reward + 1.5
            elif distance < 0.015:
                reward = reward + 1.0
            elif distance < 0.03:
                reward = reward + 0.5

            return reward
            


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robot.forwardKinmViaPositions(action / self.damping)
        self.n_of_tries_for_point += 1
        obs = self._get_obs()

        done = False
        if error_test(self.robot.p_e, self.goal):
            info = {
                'is_success': np.float32(1.0),
            }
        else:
            info = {
                'is_success': np.float32(0.0),
            }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        self.episode_score += reward
        return obs, reward, done, info




    def reset(self):
        # TODO: initialize robot joints state to a random (but valid (in joint range)) initial state
        self.episode_score = 0
        self.n_of_points_done += 1
        self.n_of_tries_for_point = 0

        # generate new point
        self.goal = np.array([random.uniform(-0.70, 0.70), random.uniform(-0.70, 0.70), random.uniform(-0.70, 0.70)])
        
        # initialize to a random starting state and check whether it makes any sense
        sensibility_check = False
        # TODO DELETE STUPID PRINT
        i = 0
        while not sensibility_check:
            i+=1
            thetas = []
            for joint in self.robot.joints:
                 thetas.append(6.28 * np.random.random() - 3.14)
            self.robot.forwardKinmViaPositions(thetas)
            if calculateManipulabilityIndex(self.robot) > 0.15:
                sensibility_check = True

        obs = self._get_obs()
        return obs





    def close(self):
        # close open files if any are there
        pass


    # various uitility functions COPIED from fetch_env

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]




    def _get_obs(self):
        thetas = []
        for joint in self.robot.joints:
            thetas.append(joint.theta)
        thetas = np.array(thetas , dtype=np.float32)
        obs = self.robot.p_e.copy()
        obs = np.append(obs, thetas)

        return {
            'observation': obs,
            'achieved_goal': self.robot.p_e.copy(),
            'desired_goal': self.goal.copy(),
        }



    def render(self, mode='human', width=500, height=500):
        try:
            self.drawingInited == False
        except AttributeError:
            self.drawingInited = False

        if self.drawingInited == False:
            plt.ion()
            self.fig = plt.figure()
            #self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax = self.fig.add_subplot(111, projection='3d')
            # these are for axes scaling which does not happen automatically
            self.ax.plot(np.array([0]), np.array([0]), np.array([1.5]), c='b')
            self.ax.plot(np.array([0]), np.array([0]), np.array([-1.5]), c='b')
            plt.xlim([-1.5,1.5])
            plt.ylim([-0.5,1.5])
            color_link = 'black'
            self.robot.initDrawing(self.ax, color_link)
            self.drawingInited = True
        self.robot.drawStateAnim()
        self.ax.set_title(str(self.n_of_tries_for_point) + 'th iteration toward goal')
        drawPoint(self.ax, self.goal, 'red')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        #return super(FetchEnv, self).render(mode, width, height)
