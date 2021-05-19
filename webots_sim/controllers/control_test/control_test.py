"""control_test controller."""

from controller import Robot, Motor, PositionSensor, GPS

from webots_api_helper_funs import *
from joint_as_hom_mat import *
#from forw_kinm import *
#from anim_func import *
import numpy as np
#import matplotlib.pyplot as plt 
#import mpl_toolkits.mplot3d.axes3d as p3
#from matplotlib.animation import FuncAnimation
#import matplotlib.colors as colr
import sys
import scipy.optimize

# angle in radians ofc
def turnAngleToPlusMinusPiRange(angle):
    return np.arcsin(np.sin(angle))


class Robot_raw:
    #def __init__(self, clamp):
    def __init__(self):
        #self.clamp = clamp
        self.clamp = 0
        self.joints = []
        fil = open('ur10e_dh_parameters_from_the_ur_site', 'r')
        params = fil.read().split('\n')
        params.pop()
        for p in params:
            p = p.split(';')
            self.joints.append(Joint(float(p[0]), float(p[1]), float(p[2]), float(p[3]), self.clamp))

        fil.close()
        self.jacobian = 0
        self.calcJacobian()

# drawing
#        self.anim_data = []
#        self.lines = []
        #initialize the plot for each line in order to animate 'em
        # each joint equates to four lines: 3 for orientation and 1 for the link
        # and each line has its x,y and z coordinate
        # thus 1 joint is: [x_hat, y_hat, z_hat, p] = [[x_hat1_x, x_hat_y, x_hat_z], [...]
        for j in range(len(self.joints)):
            x_hat = self.joints[j].HomMat[0:3,0]
            y_hat = self.joints[j].HomMat[0:3,1]
            z_hat = self.joints[j].HomMat[0:3,2]
            p = self.joints[j].HomMat[0:3,3]
# drawing
#            self.anim_data += [[x_hat, y_hat, z_hat, p]]
#            line_x, = plt.plot([],[]) 
#            line_y, = plt.plot([],[])
#            line_z, = plt.plot([],[])
#            line_p, = plt.plot([],[], 'g')
#            self.lines += [[line_x, line_y, line_z, line_p]]

    def saveConfToFile(self):
        fil = open('robot_parameters', 'r')
        params = fil.read().split('\n')
        fil.close()
        fil = open('robot_parameters', 'w')
        params.pop()
        back_to_string = ''
        for i in range(len(params)):
            params[i] = params[i].split(';')
            params[i][1] = self.joints[i].theta
            for j in range(4): #n of d-h params
                back_to_string += str(params[i][j])
                if j != 3:
                    back_to_string += ';'
                else:
                    back_to_string += '\n'
        fil.write(back_to_string) 
        fil.close()


# the jacobian is calulated for the end effector in base frame coordinates
#   TODO: the base frame ain't aligned with world coordinate frame!!!
#       ==> the only offset is on the z axis tho
#   TODO: joint positions are to be sensed from sensors, not read from as an internal state!
# it could also be calculated numerically (it's a derivative), but this is analytic, hence preferable
    def calcJacobian(self):
        z_is = [np.array([0,0,1])]
        p_is = [np.array([0,0,0])]
        toBase = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        self.p_e = np.array([0.,0.,0.])
        
        for j in range(len(self.joints)):
            toBase = toBase @ self.joints[j].HomMat
            z_is.append(toBase[0:3, 2])
            p_is.append(toBase[0:3, 3])
        p_e = p_is[-1]
        self.p_e = p_is[-1]
        jac = np.array([0,0,0,0,0,1]).reshape(6,1)
# we doin only revolute joints still 

        for j in range(len(self.joints)):
            j_p = np.cross(z_is[j], p_e - p_is[j]).reshape(3,1)
            j_o = z_is[j].reshape(3,1)
            j_i = np.vstack((j_p, j_o))
            jac = np.hstack((jac, j_i))
        self.jacobian = jac[0:6,1:]
#        print("jacobian incoming")
#        print(self.jacobian)
        self.jac_tri = self.jacobian[0:3,:]
            

    def drawState(self, ax, color_link):
        toBase = [np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])]
        drawOrientation(ax, toBase[-1][0:3,0:3], np.zeros((3,)))
        for j in range(len(self.joints)):
            toBase.append(toBase[-1] @ self.joints[j].HomMat)
            drawOrientation(ax, toBase[-1][0:3,0:3], toBase[-1][0:3,3])
            drawVector(ax, -1* ( toBase[-2][0:3,3] - toBase[-1][0:3,3] ), toBase[-2][0:3,3],color_link)

   # the thetas are to be read from position sensors 
    def forwardKinm(self, sensors):
        thetas = np.array(readJointState(sensors))
# potential clamping for joint rotation limits
        for i in range(len(thetas)):
            if self.clamp == 1:
                self.joints[i].rotate_numerically(clampTheta(thetas[i]), self.clamp)
            else:
                self.joints[i].rotate_numerically(thetas[i], self.clamp)
        self.calcJacobian()

    def eePositionAfterForwKinm(self, thetas):
        joints2 = self.joints
        for i in range(len(thetas)):
            joints2[i].rotate(joints2[i].theta + thetas[i])

        toBase = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        
        for j in range(len(joints2)):
            toBase = toBase @ joints2[j].HomMat
        p_e = toBase[0:3,3]
        return p_e

    def printJacobianForSomeAngles(self, thetas):
        self.forwardKinm(thetas)
#        print("JACOBIAN")
#        print(self.jacobian)
        print("ee position")
        print(self.p_e)








# this way of doing FK makes no sense in the context of simulations,
#   since, the moving is the done by something outside of the controller program.
# Thus, it makes no sense to keep joint positions, velocities or whatever
#   as internal states => they are meant to be sensed from the environment.
# Thus, FK in the simulator (and reality) should send the desired commands to the robot
#   and do nothing more!
# But how do we then know the state? After all, we do not care for FK, we care about
#   getting to the desired position (or following a set of positions).
# The solution is to decouple commands and states. In concrete terms, 
#   that means having different functions/methods for each task.
# TODO: remake FK function so that it just applies commands to the motors and nothing more.
#   ==> that means having a fun for positions and a fun for velocities.
# Furthermore, that means having 2 sensor-reading functions: one for position (in joint space)
#   and one for velocities (also in joint space). 
# With that, we have commands and we have our internal state.
# Again, we do actually care about any of this, we care about how the manipulator works
#   in the physical world. That is the outcome of the above and can be only monitored,
#   and even that is a problem (you got sensor noise the real world).

    def forwardKinmViaPositions(self, thetas, simulated_robot_motors, sensors):
# potential clamping for joint rotation limits
#   ==> ur10e does not have joint limits, but other robots might
        #np.arcsin(np.sin(np.array(thetas)))
        for i in range(len(thetas)):
            if self.clamp == 1:
                #TODO write this out
                self.joints[i].rotate(clampTheta(self.joints[i].theta + thetas[i]), self.clamp)
            else:
                simulated_robot_motors[i].setPosition(np.arcsin(np.sin(np.array(self.joints[i].theta + thetas[i]))))
                # here you have to read the values and then "rotate"
                #self.joints[i].rotate_numerically(self.joints[i].theta + thetas[i], self.clamp)
                #self.joints[i].rotate_numerically(thetas[i], self.clamp)

        #HERE YOU NEED TO PLANT THE FUNCTION WHICH CALCS JACOBIAN FROM AFTER READING FROM SENSORS
        #self.calcJacobian()
        self.forwardKinm(sensors)




    def forwardKinmViaVelocities(self, thetas, simulated_robot_motors):
# potential clamping for joint velocity limits
#   ==> done in the ik method!
        for i in range(len(thetas)):
            if self.clamp == 1:
                # TODO
                self.joints[i].rotate(clampTheta(self.joints[i].theta + thetas[i]), self.clamp)
            else:
                simulated_robot_motors[i].setVelocity(thetas[i])
                self.joints[i].rotate_numerically(thetas[i], self.clamp)
        #HERE YOU NEED TO PLANT THE FUNCTION WHICH CALCS JACOBIAN FROM AFTER READING FROM SENSORS
        self.calcJacobian()




# in webots, i get this for free with a position sensor
#   ==> there is no end effector position sensor!!!
#       ==> it's not in reality and it is not in the simulation!
# there are joint position sensors and you can calculate ee_pos from those,
#   i.e. just calc the jacobian and extract it from there to base frame coordinates
    def eePositionAfterForwKinm(self, thetas):
        joints2 = self.joints
        for i in range(len(thetas)):
            joints2[i].rotate(joints2[i].theta + thetas[i])

        toBase = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        
        for j in range(len(joints2)):
            toBase = toBase @ joints2[j].HomMat
        p_e = toBase[0:3,3]
        return p_e
        
    
    def bruteAnimForwKinm(self, ax, thetas):
        shots = np.linspace(0,1,20)
        for shot in shots:
            for i in range(len(thetas)):
                self.joints[i].rotate(self.joints[i].theta + thetas[i] * shot)
            self.drawState(ax, 'bisque')
        self.calcJacobian()
        self.drawState(ax, 'b')

    
    # remake this into the FuncAnimation update function
    # divide each angle from the current to desired angle in the same number of steps
    # then for each joint and each of its steps calculate the new position,
    # and append it in the appropriate "line" list
# there is no set_data() for 3dim, so we use set_3_properties() for the z axis
def forwardKinAnim(frame, r, thetas, n_of_frames, ax):
    thetas_sliced = []
#    print("frame")
 #   print(frame)
    for j in range(len(r.joints)):
        # i have no idea why but the sum of frame(i)/n_of_frames adds up to 0.5
        # we need it to add to to 1 so that in total the manipulator rotates by the specified angles
        thetas_sliced.append(thetas[j] * (2 * frame / n_of_frames))
    r.forwardKinm(thetas_sliced)
    toBase = [np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])]
    x_hat_dat =[[0],[0],[0]]
    y_hat_dat =[[0],[0],[0]]
    z_hat_dat =[[0],[0],[0]]
    p_dat = [[0],[0],[0]]
    for j in range(len(r.joints)):
        toBase.append(toBase[-1] @ r.joints[j].HomMat)
        orientation = toBase[-1][0:3,0:3] 
        x_hat = orientation[0:3,0]  + toBase[-1][0:3,3]
        y_hat = orientation[0:3,1] + toBase[-1][0:3,3]
        z_hat = orientation[0:3,2] + toBase[-1][0:3,3]
        p = (-1* ( toBase[-2][0:3,3] - toBase[-1][0:3,3] ) + toBase[-2][0:3,3])
        for i in range(3):
            x_hat_dat[i].append(x_hat[i])
            y_hat_dat[i].append(y_hat[i])
            z_hat_dat[i].append(z_hat[i])
            p_dat[i].append(p[i])
#            r.lines[j][0].set_data(x_hat_dat[0], x_hat_dat[1])
#            r.lines[j][0].set_3d_properties(x_hat_dat[2])
#            r.lines[j][1].set_data(y_hat_dat[0], y_hat_dat[1])
#            r.lines[j][1].set_3d_properties(y_hat_dat[2])
#            r.lines[j][2].set_data(z_hat_dat[0], z_hat_dat[1])
#            r.lines[j][2].set_3d_properties(z_hat_dat[2])
            r.lines[j][3].set_data(p_dat[0], p_dat[1])
            r.lines[j][3].set_3d_properties(p_dat[2])
#    print(frame / frame_length)
    if frame == 1:
        r.drawState(ax, 'r')

# they are arranged in the same order they are given in the parameters file
# i don't care about names, i just want an ordered list - preserving order is then easy

# hardcoded for all joints
# of course each joint can have its own limit
def clampVelocity(del_thet):
    for indeks in range(len(del_thet)):
        if del_thet[indeks] > 3.0:
            del_thet[indeks] = 3.0 
        
        if del_thet[indeks] < -3.0:
            del_thet[indeks] = -3.0 
    return del_thet
# r is the robot
# t is the target position


# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

print("eto me")

#hand_motors = [0] * 3
#ur_motors = [0] * 3

#hand_motors_0 = robot.getMotor("elbow_joint")
#shoulder_joint = robot.getMotor("shoulder_pan_joint")
#shoulder_joint.setPosition(1.0)
#shoulder_joint.setVelocity(2.0)


#enable gps
gps = robot.getGPS("ee_gps")
gps.enable(10)





# Main loop:
# - perform simulation steps until Webots is stopping the controller
#time = 0
t = np.array([-0.7,-0.6,0.29])
iter_num = 0
motors = getAllMotors(robot)
initializeMotorsForPosition(motors)
r = Robot_raw()
sensors = getAndInitAllSensors(robot)
#time.sleep(3)
#motors[0].setPosition(float('inf'))

for motor in motors:
    motor.setVelocity(float('inf'))
   
for motor in motors:
        motor.setPosition(0.0)
#motors[-2].setPosition(-1.57)
#motors[-1].setPosition(3.14)
while robot.step(timestep) != -1:
    
    iter_num += 1
    #motors[-1].setPosition(3.14)
    
    if iter_num % 57 == 0:
        motors[0].setPosition(iter_num / 100)
    #motors[1].setPosition(-1.57)

    #motors[0].setVelocity(-1)
   # if(iter_num >= 50):
    #    motors[0].setPosition(0.3)
#        motors[0].setPosition(2.5)
#        for i in range(len(motors) - 2):
#            motors[i].setPosition(0)
#            motors[i + 1].setPosition(1.6)
#            motors[i + 2].setPosition(-1.6)
#    else:
#        if(iter_num == 50):
#            motors[0].setPosition(-2.5)
#
#        if(iter_num == 100):
#            for m in range(len(motors)):
#                motors[i].setPosition(-0.5)
#
#        if(iter_num == 150):
#            for m in range(len(motors)):
#                motors[i].setPosition(0.5)



#    print("r.p_e")
#    print(r.p_e)
#    print("gps.getValues()")
#    print(gps.getValues())
#    print("target")
#    print(t)
#    ee_pos_gps = gps.getValues()
#    ee_pos_gps[0] = -1 * ee_pos_gps[0]
#    z_cp = ee_pos_gps[1]
#    ee_pos_gps[1] = ee_pos_gps[2]
#    ee_pos_gps[2] = z_cp
##do gps
#
#
#
#  #  motors[1].setPosition(-2.2)
#    current_joint_positions = readJointState(sensors)
#
#    e = t - np.array(ee_pos_gps)
##    e = t - r.p_e
#        
##    all_thetas = []
#    last_it = iter_num
##    for i in range(iter_num):
#    error = np.sqrt(np.dot(e,e))
#    
#    print("position error:", error)
#    num = np.dot(e, r.jac_tri @ r.jac_tri.T @ e)
#    den = np.dot(r.jac_tri @ r.jac_tri.T @ e, r.jac_tri @ r.jac_tri.T @ e)
#    alpha = num / den
#    del_thet = alpha *3 * r.jac_tri.T @ e
#
## clamping for joint rotation limits
#    del_thet = clampVelocity(del_thet)
#    print("del_thet")
#    print(del_thet)
##    print("current_joint_positions")
##    print(current_joint_positions)
##    r.printJacobianForSomeAngles(current_joint_positions)
#  #  setMotorSpeeds(motors, del_thet)
#    r.forwardKinmViaPositions(del_thet, motors, sensors)
#
## if you want a damping constant other than alpha
##        del_thet = 0.011 * r.jac_tri.T @ e
##############################################
## NOTE
## forward kinm actually goes away
## that's because now you'll just use the webots robot velocities
##   and it's ee position
## also, most likely, you'll need to calc the jacobian and so on
#   from the position sensors on the robot
#   because those are the actual values
# that's a pr
