#https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
#https://spinningup.openai.com/en/latest/algorithms/vpg.html
#using 2 layers never seems to work
#relu trains a lot faster than tanh (200 vs 800 episodes)
#smaller learning rate (like the adam default of 0.001) does not work
#hidden dim of 128 has a lot less catastrophic forgetting compared to 32 dim
#dropout causes forgetting
#bias = False in linear layers seem to make zero difference
#don't have to normalize returns, but speeds up training
#sum of discounted return does not work, return-to-go does 
#https://www.quora.com/Whats-the-difference-between-Reinforce-and-Actor-Critic
#https://www.quora.com/What-is-the-difference-between-policy-gradient-methods-and-actor-critic-methods

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gym

seed = 1234
hidden_dim = 128
learning_rate = 0.01
n_episodes = 1000
gamma = 0.99

env = gym.make('CartPole-v1')

env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state):
        return self.fc(state)

class Transition:
    def __init__(self, log_prob_action, reward):
        
        self.log_prob_action = log_prob_action
        self.reward = reward

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

policy = MLP(input_dim, hidden_dim, output_dim)

print(policy)

policy_optimizer = optim.Adam(policy.parameters(), lr = learning_rate)

episode_rewards = []

for episode in tqdm(range(n_episodes)):

    transitions = []
    episode_reward = 0

    state = env.reset()

    while True:

        state = torch.FloatTensor(state).unsqueeze(0)

        action_pred = policy(state)

        action_prob = F.softmax(action_pred, dim = -1)

        dist = distributions.Categorical(action_prob)

        action = dist.sample()

        log_prob_action = dist.log_prob(action)
        
        next_state, reward, done, _ = env.step(action.item())

        transitions.append(Transition(log_prob_action, reward))

        episode_reward += reward

        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            break

    returns = []
    R = 0

    for t in reversed(transitions):
        R = t.reward + gamma * R
        returns.insert(0, R)

    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / returns.std()

    log_prob_actions = torch.cat([t.log_prob_action for t in transitions])

    loss = - (returns * log_prob_actions).sum()

    policy_optimizer.zero_grad()

    loss.backward()

    policy_optimizer.step()    

fig = plt.figure()
plt.plot(episode_rewards)
fig.savefig('pg-rewards.png')