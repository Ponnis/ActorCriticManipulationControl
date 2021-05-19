#from anim_func import *
import numpy as np
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colr
import sys
from inv_kinm_for_trajectory import *
from forw_kinm import *


vec_start = np.array([5.,-5.,4.])
vec_end = np.array([5.,5.,4.])
steps = np.arange(0.0, 1.0, 0.1)

xs = vec_start[0] + steps * (vec_end[0] - vec_start[0])
ys = vec_start[1] + steps * (vec_end[1] - vec_start[1])
zs = vec_start[1] + steps * (vec_end[2] - vec_start[2])

print(ys)
#clamp = int(sys.argv[3]) 
#method = sys.argv[2]
all_thetas = []
n_of_it = 20
    

r1 = Robot(0)
fig = plt.figure()
ax = p3.Axes3D(fig)
it = 0
iter_text = ax.text(0, 15, 20, "Iteration: 0", ha='center', va='center')
n_of_it = 80

r1.drawState(ax, 'b')

for i in range(len(xs)):
    t = [xs[i], ys[i], zs[i]]
    all_thetas, n_of_it = invKinm_dampedSquares(ax, r1, t, n_of_it)

# parse the cli input
#for i in range(len(t)):
#    t[i] = float(t[i])
#method = sys.argv[2]

#if method == 'T':    
#    all_thetas, n_of_it = invKinm_Jac_T(ax, r1, t, n_of_it)
#    r1.saveConfToFile()
#
#if method == 'P':    
#    print("t prije poziva")
#    print(t)
#    all_thetas, n_of_it = invKinm_PseudoInv(ax, r1, t, n_of_it)
#    r1.saveConfToFile()
#
#if method == 'S':    
#    all_thetas, n_of_it = invKinm_dampedSquares(ax, r1, t, n_of_it)
#    r1.saveConfToFile()
#if method == 'G':    
#    all_thetas, n_of_it = invKinmGradDesc(ax, r1, t, n_of_it)
#    r1.saveConfToFile()
##t = np.array([-10.3,-13.0,6.3])
#
#n_of_frames = 40
#ani = FuncAnimation(fig, invKinmAnim, frames=np.linspace(0, 1, n_of_frames), fargs=(r2, \
                       #all_thetas, n_of_frames, n_of_it, ax),repeat=False, blit=False)
#                       all_thetas, n_of_frames, n_of_it, ax),repeat=False, blit=False)



# you cant manually set the size of z axis because that functionality is unfortunately not implemented rn
# so let's just plot two dots which will force autoscale to work
ax.plot(np.array([0]), np.array([0]), np.array([20]), c='b')
ax.plot(np.array([0]), np.array([0]), np.array([-20]), c='b')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.xlim([-20,20])
plt.ylim([-10,20])
print("###############################################################")
print("###############################################################")
plt.show()



