import numpy as np
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colr

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
