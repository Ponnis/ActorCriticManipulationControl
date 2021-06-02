import numpy as np
import pickle
import matplotlib.pyplot as plt

file = open('results_7_big', 'rb')
#file = open('results_8_big', 'rb')
results = pickle.load(file)
file.close()


file = open('./results_big_net_only', 'rb')
#file = open('results_8_big', 'rb')
results_extra = pickle.load(file)
file.close()

results.update(results_extra)

nExperiments = results['meta']['nExperiments']
nSteps = results['meta']['nSteps']

algs = list(results.keys())
algs.remove('meta')
returns = []
returns_stds = []
successes = []
final_dists = []
final_dists_stds = []
smallest_eigenvals = []
smallest_eigenvals_stds = []
manip_indexes = []
manip_indexes_stds = []
singularities = []


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

    # percentage of singularities
    singularities.append((results[alg]['singularities'] / nExperiments) * 100)



# returns
plt.bar(algs, returns, yerr=returns_stds, ecolor='black', capsize=10)
plt.autoscale()
plt.title('Average returns')
plt.ylabel('return')
plt.savefig('./result_graphs/returns.png', dpi=600)
plt.show()

# successes
plt.bar(algs, successes)
plt.title('Success percentage')
plt.ylabel('percentage')
plt.savefig('./result_graphs/successes.png', dpi=600)
plt.show()

# final distance
plt.bar(algs, final_dists, yerr=final_dists_stds, ecolor='black', capsize=10)
plt.title('Average final distance to goal')
plt.ylabel('distance to goal in meters')
plt.savefig('./result_graphs/final_distances.png', dpi=600)
plt.show()

# smallest eigenvalue
plt.bar(algs, smallest_eigenvals, yerr=smallest_eigenvals_stds, ecolor='black', capsize=10)
plt.title('Average smallest eigenvalue of the manipulability ellipsoid')
plt.ylabel('smallest eigenvalue')
plt.savefig('./result_graphs/smallest_eigenvals.png', dpi=600)
plt.show()

# manipulability index
plt.bar(algs, manip_indexes, yerr=manip_indexes_stds, ecolor='black', capsize=10)
plt.title('Average manipulability index')
plt.ylabel('manipulability index')
plt.savefig('./result_graphs/manip_indexes.png', dpi=600)
plt.show()


# singularities
plt.bar(algs, singularities)
plt.title('Percentage of runs running into singularities')
plt.ylabel('percentage')
plt.savefig('./result_graphs/singularities.png', dpi=600)
plt.show()

