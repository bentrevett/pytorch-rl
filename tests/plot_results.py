import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

experiments = os.listdir('results')

os.makedirs('figures', exist_ok=True)

fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for experiment in experiments:
    results = np.loadtxt(f'results/{experiment}')
    mean_results = np.mean(results, axis=0)
    ax.plot(mean_results, label=f'{experiment[:-4]}') #remove the .txt

plt.title('Mean Episode Rewards', fontsize=15)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Reward', fontsize=15)
plt.legend()
plt.grid()
fig.savefig('figures/episode-rewards')
plt.clf()

fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for experiment in experiments:
    results = np.loadtxt(f'results/{experiment}')
    cum_results = np.mean(np.cumsum(results, axis=1), axis=0)
    ax.plot(cum_results, label=f'{experiment[:-4]}') #remove the .txt

plt.title('Cumulative Episode Rewards', fontsize=15)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Cumulative Reward', fontsize=15)
plt.legend()
plt.grid()
fig.savefig('figures/cumulative-episode-rewards')
plt.clf()