import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

from tqdm import tqdm
import numpy as np
import gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v1')
parser.add_argument('--n_seeds', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--episodes', type=int, default=500)
parser.add_argument('--discount_factor', type=float, default=0.99)
parser.add_argument('--init', type=str, default='none')
parser.add_argument('--activation', type=str, default='relu')
args = parser.parse_args()

assert args.init in ['none', 'xavier-normal', 'xavier-uniform']
assert args.activation in ['relu', 'tanh']

env = gym.make(args.env)

assert isinstance(env.observation_space, gym.spaces.Box)
assert isinstance(env.action_space, gym.spaces.Discrete)

seeds = [s for s in range(args.n_seeds)]
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, activation):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        activations = {'relu': F.relu, 'tanh': torch.tanh}
        self.activation = activations[activation]

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.fc_2(x)
        return x

def init_weights_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def init_weights_uniform(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def init_bias_zero(m):
    if type(m) == nn.Linear:
        m.bias.data.fill_(0)

def train(env, policy, optimizer, discount_factor):
    
    log_prob_actions = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        action_preds = policy(state)
        
        action_probs = F.softmax(action_preds, dim = -1)
                
        dist = distributions.Categorical(action_probs)

        action = dist.sample()
        
        log_prob_action = dist.log_prob(action)
        
        state, reward, done, _ = env.step(action.item())

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)

        episode_reward += reward

    log_prob_actions = torch.cat(log_prob_actions)

    returns = calculate_returns(rewards, discount_factor)

    loss = update_policy(returns, log_prob_actions, optimizer)

    return loss, episode_reward

def calculate_returns(rewards, discount_factor, normalize = True):
    
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns

def update_policy(returns, log_prob_actions, optimizer):
    
    returns = returns.detach()

    loss = - (returns * log_prob_actions).mean() #see https://arxiv.org/pdf/1709.00503.pdf eqn 2
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    return loss.item()

experiment_rewards = np.zeros((len(seeds), args.episodes))

for seed in seeds:

    env = gym.make(args.env)

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy = MLP(input_dim, args.hidden_dim, output_dim, args.dropout, args.activation)

    if args.init == 'xavier-uniform':
        policy.apply(init_weights_uniform)
    elif args.init == 'xavier-normal':
        policy.apply(init_weights_normal)
    elif args.init == 'none':
        policy.apply(init_bias_zero)
    else:
        raise ValueError('Initialization scheme not recognized!')

    optimizer = optim.Adam(policy.parameters(), lr = args.lr)

    episode_rewards = []

    for episode in tqdm(range(args.episodes)):

        loss, episode_reward = train(env, policy, optimizer, args.discount_factor)

        episode_rewards.append(episode_reward)

        experiment_rewards[seed][episode] = episode_reward

os.makedirs('results', exist_ok=True)

np.savetxt(f'results/mc_vpg_{args.hidden_dim}_{args.dropout}_{args.lr}_{args.init}_{args.activation}.txt', experiment_rewards, fmt='%d')