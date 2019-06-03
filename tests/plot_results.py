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

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Episode')
plt.ylabel('Reward')
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

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.grid()
fig.savefig('figures/cumulative-episode-rewards')
plt.clf()