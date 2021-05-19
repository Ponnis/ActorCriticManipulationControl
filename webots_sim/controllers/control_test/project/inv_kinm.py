from forw_kinm import *
from anim_func import *
import numpy as np
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colr
import sys
import scipy.optimize


# hardcoded for all joints
# of course each joint can have its own limit
def clampVelocity(del_thet):
    for indeks in range(len(del_thet)):
        if del_thet[indeks] > 2.0:
            del_thet[indeks] = 2.0 
        
        if del_thet[indeks] < -2.0:
            del_thet[indeks] = -2.0 
    return del_thet
# r is the robot
# t is the target position

def invKinm_Jac_T(ax, r, t, iter_num):
    # drawing
#    ax.set_title("Jacobian transpose method" )
#    drawPoint(ax, t, 'r')
    e = t - r.p_e
    all_thetas = []
    last_it = iter_num
    for i in range(iter_num):
        error = np.sqrt(np.dot(e,e))
        
        print("position error:", error)
        num = np.dot(e, r.jac_tri @ r.jac_tri.T @ e)
        den = np.dot(r.jac_tri @ r.jac_tri.T @ e, r.jac_tri @ r.jac_tri.T @ e)
        alpha = num / den
        del_thet = alpha * r.jac_tri.T @ e

# clamping for joint rotation limits
        del_thet = clampVelocity(del_thet)

# if you want a damping constant other than alpha
#        del_thet = 0.011 * r.jac_tri.T @ e
        r.forwardKinm(del_thet)
    # drawing
#        all_thetas.append(del_thet)
        if error < 0.1:
            last_it = i
            print("we have finished after,", i,"steps")
            print(r.p_e)
            break
        e = t - r.p_e
        if i == iter_num -1:
            print(r.p_e)
    return all_thetas, last_it

# e buraz uopce nisi iskoristio nullspace ovdje za ista lel
def invKinm_PseudoInv(ax, r, t, iter_num):
    # drawing
#    ax.set_title("Jacobian pseudoinverse method" )
#    drawPoint(ax, t, 'r')
    e = t - r.p_e
    all_thetas = []
    last_it = iter_num
    for i in range(iter_num):
        error = np.sqrt(np.dot(e,e))
        print("position error:", error)

        psedo_inv = np.linalg.pinv(r.jac_tri)
        del_thet = psedo_inv @ e
# we can add any nulspace vector to del_thet
# and given the constraints, we should implement some sort of a comfort function
# the min and max theta for each angle are hardcoded, but that will be changed 
# they are hardcoded to +/- pi # 3/4 with center at 0
# thus for all i q_iM - q_im = pi * 6/4
# the added q_0 must be left multiplyed by (np.eye(n) - np.linalg.pinv(r.jac_tri) @ r.jac_tri)
# we take into account the current theta (punish more if closer to the limit)
# the formula is 3.57 in siciliano 
        theta_for_limits = []
        for k in range(len(del_thet)):
            theta_for_limits.append( (-1/len(del_thet)) * (r.joints[k].theta / (np.pi * 1.5)))
        theta_for_limits = np.array(theta_for_limits)

# just exercising to see whether anything makes sense
#        theta_for_limits = []
#        for t in range(len(del_thet)):
#            theta_for_limits.append(0.1) 
#        theta_for_limits = np.array(theta_for_limits)

        del_thet += (np.eye(len(del_thet)) - psedo_inv @ r.jac_tri) @ theta_for_limits

        del_thet = clampVelocity(del_thet)

    # drawing
#        all_thetas.append(del_thet)
        r.forwardKinm(del_thet)
        if error < 0.01:
            print("we have finished after,", i,"steps")
            print(r.p_e)
            last_it = i
#            r.drawState(ax, 'r')
            break
        e = t - r.p_e
        if i == iter_num -1:
            print(r.p_e)
    return all_thetas, last_it


def invKinm_dampedSquares(ax, r, t, iter_num):
    # drawing
#    ax.set_title("Damped squares method" )
#    drawPoint(ax, t, 'r')
    e = t - r.p_e
    all_thetas = []
    last_it = iter_num
    lamda = 1.5
    iden = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    for i in range(iter_num):
        error = np.sqrt(np.dot(e,e))
        print("position error:", error)
        del_thet = r.jac_tri.T @ np.linalg.inv(r.jac_tri @ r.jac_tri.T + lamda**2 * iden) @ e

        del_thet = clampVelocity(del_thet)

        # let's try to use the calculation which uses svd
        # the derivation is in the iksurvey section 6
        # the final equation is (11) and it calculates all left of e in above del_thet

        # something is wrong here and i have  no idea what
        m = 3
        n = len(r.jac_tri[0,:])
        svdd = np.zeros((n, m))
        svd = np.linalg.svd(r.jac_tri) # the output is a list of 3 matrices: U, D and V
        # important note: D is not returned as a diagonal matrix, but as the list of diagonal entries
        for s in range(m): # 3 is the maximum rank of jacobian
            svdd = svdd + (svd[1][s] / (svd[1][s] ** 2 + lamda ** 2)) * svd[2][:,s].reshape(n,1) @ svd[0][:,s].reshape(1, m)


#        del_thet = svdd @ e
#        del_thet = clampVelocity(del_thet)

    # drawing
#        all_thetas.append(del_thet)
        r.forwardKinm(del_thet)
#        r.drawState(ax, 'gray')
#        if i == iter_num - 1:
#            r.drawState(ax, 'r')
        if error < 0.01:
            print("finished after,", i,"steps")
            print(r.p_e)
#            r.drawState(ax, 'r')
            last_it = i
            break
        e = t - r.p_e
        if i == iter_num -1:
            print(r.p_e)
    return all_thetas, last_it



# gradient desent
def invKinmBadGradDesc(ax, r, t, iter_num):
    # drawing
    #ax.set_title("Gradient desent" )
    #drawPoint(ax, t, 'r')
    e = t - r.p_e
    all_thetas = []
    last_it = iter_num
    # move by 0.01
    step = 0.01
    desent_speed = 0.4
    for i in range(iter_num):
        error = np.sqrt(np.dot(e,e))
        print("position error:", error)
        # we aproximate the gradient first approximating the partial derivative of each joint contribution
        # we do that from definition of the derivative
        thetas_after_step = []
        for th in range(len(r.joints)):
            thetas_after_step.append(r.joints[th].theta + step)
  
        gradient_coefs = [] 
        for th in range(len(r.joints)):
            move_th = []
            for part in range(len(r.joints)):
                if part == th:
                    move_th.append(r.joints[th].theta + thetas_after_step[th]) 
                else:
                    move_th.append(r.joints[th].theta)
            p_e_after_move_th = r.eePositionAfterForwKinm(move_th)
            e2 = t - p_e_after_move_th
            error2 = np.sqrt(np.dot(e2,e2))
            partial_th = (error2 - error) / step
            gradient_coefs.append(partial_th)

        del_thet = []
        # ok so now we have our gradients coefficients in a list
        for th in range(len(r.joints)):
            gradient_coefs[th] *= desent_speed
            del_thet.append(r.joints[th].theta * (gradient_coefs[th] - 1))
 
        del_thet = np.array(del_thet)


        del_thet = clampVelocity(del_thet)
        r.forwardKinm(del_thet)
    # drawing
    #    all_thetas.append(del_thet)
        if error < 0.1:
            last_it = i
            print("we have finished after,", i,"steps")
            print(r.p_e)
            break
        e = t - r.p_e
        if i == iter_num -1:
            print(r.p_e)
    return all_thetas, last_it



def invKinmGradDesc(ax, r, t, iter_num):
    
    def getEEPos(thetas, r, t):
        p_e = r.eePositionAfterForwKinm(thetas)
        e = t - p_e
        error = np.sqrt(np.dot(e,e))
        return error
    
    def toOptim(thetas):
        return np.sqrt(np.dot(thetas, thetas))
    # drawing
    #ax.set_title("Gradient desent" )
    #drawPoint(ax, t, 'r')
    e = t - r.p_e
    all_thetas = []
    last_it = iter_num
    lb = []
    ub = []

    def constraint(r, e):
        # jac_tri @ del_thet must be equal to e
        return scipy.optimize.LinearConstraint(r.jac_tri, e, e)


    for bo in range(len(r.joints)):
        lb.append(-2.0)
        ub.append(2.0)
    bounds = scipy.optimize.Bounds(lb, ub)

    for i in range(iter_num):
        error = np.sqrt(np.dot(e,e))
        print("position error:", error)
        thetas_start = []
        for th in range(len(r.joints)):
            thetas_start.append(r.joints[th].theta)
        thetas_start = np.array(thetas_start)

        lin_constraint = constraint(r, e)
        if (r.clamp == 1):
            res = scipy.optimize.minimize(toOptim, thetas_start, method='SLSQP', constraints=lin_constraint, bounds=bounds)
        else:
            res = scipy.optimize.minimize(toOptim, thetas_start, method='SLSQP', constraints=lin_constraint)
#        res = scipy.optimize.minimize(getEEPos, thetas_start, args=(r,t), method='SLSQP', constraints=lin_constraint, bounds=bounds)
#        res = scipy.optimize.minimize(toOptim, thetas_start, method='CG', bounds=bounds)
        # without constraints it returns some crazy big numbres like 10**300 or sth
        # so something is seriously wrong there
        del_thet = []
        for bla in range(len(res.x)):
            del_thet.append(float(res.x[bla]))
#            del_thet.append(res.x[bla] - 0.01)
#        for bla in range(len(res.x)):
#            del_thet.append(float(res.x[bla]))
#            del_thet[bla] += 0.01
#            print(del_thet[bla])
#        print("del_thet")
#        print(del_thet)
 
#        del_thet = np.array(del_thet)
        del_thet = clampVelocity(del_thet)
        r.forwardKinm(del_thet)
    #    all_thetas.append(del_thet)
        if error < 0.1:
            last_it = i
            print("we have finished after,", i,"steps")
            print(r.p_e)
            break
        e = t - r.p_e
        if i == iter_num -1:
            print(r.p_e)
    return all_thetas, last_it


def invKinmAnim(frame, r, thetas, n_of_frames, n_of_it, ax):
    thetas_sliced = []
    global it
    global iter_text
    iter_text.set_text("Iteration: " + str(it // 40))
    for i in range(n_of_it):
        thetas_i_slice = []
        for j in range(len(r.joints)):
            thetas_i_slice.append(thetas[i][j] * (2 * frame / n_of_frames))
        thetas_sliced.append(thetas_i_slice)

    for i in range(n_of_it):
        it += 1
        r.forwardKinm(thetas_sliced[i])
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
#            p = (-1* ( toBase[-2][0:3,3] - toBase[-1][0:3,3] ) + toBase[-2][0:3,3])
            p =  toBase[-1][0:3,3] 
            for i in range(3):
                x_hat_dat[i].append(x_hat[i])
                y_hat_dat[i].append(y_hat[i])
                z_hat_dat[i].append(z_hat[i])
                p_dat[i].append(p[i])
#                r.lines[j][0].set_data(x_hat_dat[0], x_hat_dat[1])
#                r.lines[j][0].set_3d_properties(x_hat_dat[2])
    #            r.lines[j][1].set_data(y_hat_dat[0], y_hat_dat[1])
    #            r.lines[j][1].set_3d_properties(y_hat_dat[2])
#                r.lines[j][2].set_data(z_hat_dat[0], z_hat_dat[1])
#                r.lines[j][2].set_3d_properties(z_hat_dat[2])
                r.lines[j][3].set_data(p_dat[0], p_dat[1])
                r.lines[j][3].set_3d_properties(p_dat[2])

    if frame == 1:
        r.drawState(ax, 'r')


    # drawing
#fig = plt.figure()
#ax = p3.Axes3D(fig)


it = 0
    # drawing
#iter_text = ax.text(0, 15, 20, "Iteration: 0", ha='center', va='center')



clamp = int(sys.argv[3])
r1 = Robot(clamp)
r2 = Robot(clamp)
#print("ok")
n_of_it = 80
#r.drawState(ax, 'green')
#r.forwardKinm(thetas1)
#r.drawState(ax, 'gray')
#small_move = np.array([-0.5,-0.5,-0.5])
#del_thetas = r.jac_tri.T @ small_move
#r.forwardKinm(del_thetas)
#r.drawState(ax, 'c')


#r1.forwardKinm(np.array([3,3,3,2]))
#r2.forwardKinm(np.array([3,3,3,2]))
    # drawing
#r1.drawState(ax, 'b')

# parse the cli input
t = sys.argv[1].split(',')
for i in range(len(t)):
    t[i] = float(t[i])
method = sys.argv[2]

if method == 'T':    
    all_thetas, n_of_it = invKinm_Jac_T(ax, r1, t, n_of_it)
    # webots remembers the state so no need
    #r1.saveConfToFile()

if method == 'P':    
    print("t prije poziva")
    print(t)
    all_thetas, n_of_it = invKinm_PseudoInv(ax, r1, t, n_of_it)
    # webots remembers the state so no need
    #r1.saveConfToFile()

if method == 'S':    
    all_thetas, n_of_it = invKinm_dampedSquares(ax, r1, t, n_of_it)
    # webots remembers the state so no need
    #r1.saveConfToFile()
if method == 'G':    
    all_thetas, n_of_it = invKinmGradDesc(ax, r1, t, n_of_it)
    # webots remembers the state so no need
    #r1.saveConfToFile()
#t = np.array([-10.3,-13.0,6.3])

    #drawing
#n_of_frames = 40
#ani = FuncAnimation(fig, invKinmAnim, frames=np.linspace(0, 1, n_of_frames), fargs=(r2, \
                       #all_thetas, n_of_frames, n_of_it, ax),repeat=False, blit=False)
#                       all_thetas, n_of_frames, n_of_it, ax),repeat=False, blit=False)

#invKinm_PseudoInv(ax, r, t, n_of_it)
#r.forwardKinm(np.array([0.1,0.,0.,0.]))
#invKinm_dampedSquares(ax, r, t, n_of_it)
#r.drawState(ax, 'green')


#thetas = []
#for i in range(len(r.joints)):
#    if i != 3:
#        thetas += [0.0]
#    else:
#        thetas += [-1.6]
#print(thetas)

#r.forwardKinm(thetas)
#r.drawState(ax, 'dimgray')
#r.bruteAnimForwKinm(ax, thetas)
#thetas = []
#for i in range(len(r.joints)):
#    thetas += [-1.2]
#r.forwardKinm(thetas)
#r.drawState(ax, 'dimgray')


# you cant manually set the size of z axis because that functionality is unfortunately not implemented rn
# so let's just plot two dots which will force autoscale to work
    #drawing
#ax.plot(np.array([0]), np.array([0]), np.array([20]), c='b')
#ax.plot(np.array([0]), np.array([0]), np.array([-20]), c='b')
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.set_zlabel("z")
#plt.xlim([-20,20])
#plt.ylim([-10,20])
#plt.show()
print("###############################################################")
print("###############################################################")
