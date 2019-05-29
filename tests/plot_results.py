import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

experiments = os.listdir('results')

os.makedirs('figures', exist_ok=True)

for experiment in experiments:
    results = np.loadtxt(f'results/{experiment}')
    mean_results = np.mean(results, axis=0)
    plt.plot(mean_results, label=f'{experiment[:-4]}') #remove the .txt

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.savefig('figures/episode-rewards')
plt.clf()

for experiment in experiments:
    results = np.loadtxt(f'results/{experiment}')
    cum_results = np.mean(np.cumsum(results, axis=1), axis=0)
    plt.plot(cum_results, label=f'{experiment[:-4]}') #remove the .txt

plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.savefig('figures/cumulative-episode-rewards')
plt.clf()