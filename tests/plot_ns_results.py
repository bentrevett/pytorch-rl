import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

ns_experiments = [e for e in os.listdir('results') if e.startswith('ns_a2c')]

os.makedirs('figures', exist_ok=True)

"""fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for experiment in ns_experiments:
    results = np.loadtxt(f'results/{experiment}')
    for r in results:
    #mean_results = np.mean(results, axis=0)
        ax.plot(r, label=f'{experiment[:-4]}') #remove the .txt

plt.title('Mean Episode Rewards', fontsize=15)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Reward', fontsize=15)
plt.legend()
plt.grid()
fig.savefig('figures/ns_episode_rewards')
plt.clf()"""

"""fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for experiment in ns_experiments:
    print(experiment)
    results = np.loadtxt(f'results/{experiment}')
    for r in results:
        cum_results = np.cumsum(r, axis=0)
        ax.plot(cum_results, label=f'{experiment[:-4]}') #remove the .txt

plt.title('Cumulative Episode Rewards', fontsize=15)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Cumulative Reward', fontsize=15)
plt.legend()
plt.grid()
fig.savefig('figures/ns_cumulative_rewards')
plt.clf()"""

"""all_results = dict()
fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for experiment in ns_experiments:

    results = np.loadtxt(f'results/{experiment}')
    avg_rew_per_ep = np.mean(results, axis=0)
    step_size = experiment.split('_')[-1]
    step_size = int(step_size[:-6])
    x = [i*step_size for i, _ in enumerate(avg_rew_per_ep)]
    ax.plot(x, avg_rew_per_ep, label=experiment)

plt.title(f'Average Reward vs Step', fontsize=15)
plt.xlabel('Step', fontsize=15)
plt.ylabel('Average Reward', fontsize=15)
#plt.legend()
plt.grid()
ax.ticklabel_format(style='plain')
fig.savefig(f'figures/ns_average_reward')
plt.clf()

assert False"""

ss = [5, 10, 25, 50, 100, 200, 500]

for s in ss:

    all_results = dict()

    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(111)

    for experiment in ns_experiments:
        if f'_{s}ns' not in experiment:
            continue
        #print(experiment)
        results = np.loadtxt(f'results/{experiment}')
        cum_results = np.cumsum(results, axis=1)
        avg_cum_results = np.mean(cum_results, axis=0)
        final_avg_cum_result = avg_cum_results[-1]
        #print(final_avg_cum_result)
        all_results[experiment[:-4]] = final_avg_cum_result

    sorted_results = [(experiment, all_results[experiment]) for experiment in sorted(all_results, key=all_results.get, reverse=True)]

    top_n_results = sorted_results[:10]

    for experiment, _ in top_n_results:

        results = np.loadtxt(f'results/{experiment}.txt')
        avg_rew_per_ep = np.mean(results, axis=0)
        step_size = experiment.split('_')[-1]
        step_size = int(step_size[:-2])
        assert step_size == s
        x = [i*step_size for i, _ in enumerate(avg_rew_per_ep)]
        ax.plot(x, avg_rew_per_ep, label=experiment)

    plt.title(f'Average Reward vs Step (Size = {s})', fontsize=15)
    plt.xlabel('Step', fontsize=15)
    plt.ylabel('Average Reward', fontsize=15)
    plt.legend()
    plt.grid()
    ax.ticklabel_format(style='plain')
    fig.savefig(f'figures/ns_average_reward_{s}')
    plt.clf()