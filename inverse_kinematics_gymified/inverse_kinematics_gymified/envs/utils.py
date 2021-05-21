import numpy as np


def error_test(p_e, target):
    e = abs(target - p_e)
    if e[0] < 0.001 and e[1] < 0.001 and e[2] < 0.001:
        return True
    else:
        return False

def goal_distance(achieved_goal, goal):
    return np.linalg.norm(goal - achieved_goal)


def calculateManipulabilityIndex(robot):
    M = robot.jac_tri @ robot.jac_tri.T
    return np.sqrt(np.linalg.det(M))

def calculateSmallestManipEigenval(robot):
    M = robot.jac_tri @ robot.jac_tri.T
    diagonal_of_svd_of_M = np.linalg.svd(M)[1]
    return diagonal_of_svd_of_M[diagonal_of_svd_of_M.argmin()]

def calculatePerformanceMetrics(robot):
    M = robot.jac_tri @ robot.jac_tri.T
    diagonal_of_svd_of_M = np.linalg.svd(M)[1]
    return {'manip_index': np.sqrt(np.linalg.det(M)),
            'smallest_eigenval': diagonal_of_svd_of_M[diagonal_of_svd_of_M.argmin()],
            'biggest_eigenval':diagonal_of_svd_of_M[diagonal_of_svd_of_M.argmax()] }

