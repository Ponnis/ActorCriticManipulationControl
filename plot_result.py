import numpy as np
import pickle
import matplotlib.pyplot as plt

file = open('results_4', 'rb')
results = pickle.load(file)
file.close()

nExperiments = results['meta']['nExperiments']
nSteps = results['meta']['nSteps']

algs = results.keys()
returns = []
returns_stds = []
successes = []
final_dists = []
final_dists_stds = []
smallest_eigenvals = []
smallest_eigenvals_stds = []
manip_indexes = []
manip_indexes_stds = []


# TODO change this so that it shows standard deviations as well
# you do it with plt.bar(algs, thing, std_thing)
# also label the y axis pls
for alg in algs:
    # returns
    returns_alg = np.array(results[alg]['returns'])
    mean = np.mean(returns_alg)
    std = np.std(returns_alg)
    returns.append(mean)
    returns_stds.append(std)

    # successes
    successes.append((results[alg]['successes'] / nExperiments) * 100)

    # final distances
    final_dists_alg = np.array(results[alg]['final_distances'])
    mean = np.mean(final_dists_alg)
    std = np.std(final_dists_alg)
    final_dists.append(mean)
    final_dists_stds.append(std)

    # smallest eigenvals
    smallest_eigenvals_alg = np.array(results[alg]['smallest_eigenvals'])
    mean = np.mean(smallest_eigenvals_alg)
    std = np.std(smallest_eigenvals_alg)
    smallest_eigenvals.append(mean)
    smallest_eigenvals_stds.append(std)


    # manip_indexes
    manip_indexes_alg = np.array(results[alg]['manip_indexes'])
    mean = np.mean(manip_indexes_alg)
    std = np.std(manip_indexes_alg)
    manip_indexes.append(mean)
    manip_indexes_stds.append(std)


# returns
plt.bar(algs, returns, yerror=returns_stds, ecolor='black')
plt.title('Average returns')
plt.ylabel('return')
plt.show()
#plt.savefig('avg_returns.png')

# successes
plt.bar(algs, successes)
plt.title('Success percentage')
plt.ylabel('percentage')
plt.show()
#plt.savefig('avg_done.png')

# final distance
plt.bar(algs, final_dists, yerror=final_dists_stds, ecolor='black')
plt.title('Average final distance to goal')
plt.ylabel('distance to goal in meters')
plt.show()

################### TODO TODO TODO ########################
# smallest eigenvalue
plt.bar(algs, smallest_eigenvals, yerror=smallest_eigenvals_stds, ecolor='black')
plt.title('Average smallest eigenvalue of the manipulability ellipsoid')
plt.ylabel('smallest eigenvalue')
plt.show()

# manipulability index
plt.bar(algs, manip_indexes, yerror=manip_indexes_stds, ecolor='black')
plt.title('Average manipulability index')
plt.ylabel('manipulability index')
plt.show()

#plt.savefig('avg_final_dist.png')
#plt.show()
