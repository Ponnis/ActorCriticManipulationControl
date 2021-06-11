import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#df = pd.read_csv('./trained_nets/her-sac-stronger-convergence-rewards-neg-plus-inverse-CLAMP_2021_06_07_19_35_09_0000--s-0/progress.csv')
df = pd.read_csv('./trained_nets/her-sac-no-manip-rewards_2021_05_25_10_55_18_0000--s-0/progress.csv')

df.plot('Epoch', 'evaluation/Average Returns')

plt.show()
