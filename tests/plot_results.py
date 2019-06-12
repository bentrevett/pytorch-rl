import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

mc_experiments = [e for e in os.listdir('results') if e.startswith('mc_')]
ns_experiments = [e for e in os.listdir('results') if e.startswith('ns_')]

os.makedirs('figures', exist_ok=True)

fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for experiment in mc_experiments:
    results = np.loadtxt(f'results/{experiment}')
    mean_results = np.mean(results, axis=0)
    ax.plot(mean_results, label=f'{experiment[:-4]}') #remove the .txt

plt.title('Mean Episode Rewards', fontsize=15)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Reward', fontsize=15)
plt.legend()
plt.grid()
fig.savefig('figures/mc_episode_rewards')
plt.clf()

fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for experiment in ns_experiments:
    results = np.loadtxt(f'results/{experiment}')
    mean_results = np.mean(results, axis=0)
    ax.plot(mean_results, label=f'{experiment[:-4]}') #remove the .txt

plt.title('Mean Episode Rewards', fontsize=15)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Reward', fontsize=15)
plt.legend()
plt.grid()
fig.savefig('figures/ns_episode_rewards')
plt.clf()

fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for experiment in mc_experiments:
    results = np.loadtxt(f'results/{experiment}')
    cum_results = np.mean(np.cumsum(results, axis=1), axis=0)
    linestyle = '-' if 'relu' in experiment else '--'
    ax.plot(cum_results, label=f'{experiment[:-4]}', linestyle=linestyle) #remove the .txt

plt.title('Cumulative Episode Rewards', fontsize=15)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Cumulative Reward', fontsize=15)
plt.legend(loc=2)
plt.grid()
fig.savefig('figures/mc_cumulative_rewards')
plt.clf()

fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for experiment in ns_experiments:
    results = np.loadtxt(f'results/{experiment}')
    cum_results = np.mean(np.cumsum(results, axis=1), axis=0)
    ax.plot(cum_results, label=f'{experiment[:-4]}') #remove the .txt

plt.title('Cumulative Episode Rewards', fontsize=15)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Cumulative Reward', fontsize=15)
plt.legend()
plt.grid()
fig.savefig('figures/ns_cumulative_rewards')
plt.clf()