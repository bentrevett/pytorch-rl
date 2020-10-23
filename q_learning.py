import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import collections
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='CartPole-v1', type=str)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--grad_clip', default=0.5, type=float)
parser.add_argument('--hid_dim', default=32, type=int)
parser.add_argument('--init', default='xavier', type=str)
parser.add_argument('--n_runs', default=5, type=int)
parser.add_argument('--n_episodes', default=1000, type=int)
parser.add_argument('--discount_factor', default=0.99, type=float)
parser.add_argument('--start_epsilon', default=1.0, type=float)
parser.add_argument('--end_epsilon', default=0.01, type=float)
parser.add_argument('--exploration_time', default=0.5, type=float)
parser.add_argument('--optim', default='adam', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
args = parser.parse_args()

name = '-'.join([f'{k}={v}' for k, v in vars(args).items()])
print(name)

import os
assert not os.path.exists('checkpoints/'+name+'_train.pt')

train_env = gym.make(args.env)
test_env = gym.make(args.env)

train_env.seed(args.seed)
test_env.seed(args.seed+1)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.n_layers == 1:
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()

            self.fc_1 = nn.Linear(input_dim, hidden_dim)
            self.fc_2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.fc_1(x)
            x = F.relu(x)
            x = self.fc_2(x)
            return x
else:
    assert args.n_layers == 2
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()

            self.fc_1 = nn.Linear(input_dim, hidden_dim)
            self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc_3 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.fc_1(x)
            x = F.relu(x)
            x = self.fc_2(x)
            x = F.relu(x)
            x = self.fc_3(x)
            return x

input_dim = train_env.observation_space.shape[0]
hidden_dim = args.hid_dim
output_dim = train_env.action_space.n

if args.init == 'xavier':
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)
else:
    assert args.init == 'kaiming'
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

def train(env, policy, optimizer, discount_factor, epsilon, device):
    
    policy.train()
    
    states = []
    actions = []
    rewards = []
    next_states = []
    done = False
    episode_reward = 0

    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    while not done:

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_pred = policy(state)
            action = torch.argmax(q_pred).item()

        next_state, reward, done, _ = env.step(action)

        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        loss = update_policy(policy, state, action, reward, next_state, done, discount_factor, optimizer)

        state = next_state
        episode_reward += reward

    return loss, episode_reward

def update_policy(policy, state, action, reward, next_state, done, discount_factor, optimizer):

    q_preds = policy(state)
    q_vals = q_preds[:, action]

    with torch.no_grad():
        q_next_preds = policy(next_state)
        q_next_vals = q_next_preds.max(1).values
        targets = reward + q_next_vals * discount_factor * done

    loss = F.smooth_l1_loss(q_vals, targets.detach())
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
    optimizer.step()
    
    return loss.item()

def evaluate(env, policy, device):
    
    policy.eval()
    
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            q_pred = policy(state)
            action = torch.argmax(q_pred).item()

        state, reward, done, _ = env.step(action)
        episode_reward += reward

    return episode_reward

n_runs = args.n_runs
n_episodes = args.n_episodes
discount_factor = args.discount_factor
start_epsilon = args.start_epsilon
end_epsilon = args.end_epsilon
exploration_time = int(args.n_episodes * args.exploration_time)

epsilons = np.linspace(start_epsilon, end_epsilon, exploration_time)

train_rewards = torch.zeros(n_runs, n_episodes)
test_rewards = torch.zeros(n_runs, n_episodes)
device = torch.device('cpu')

for run in range(n_runs):
    
    policy = MLP(input_dim, hidden_dim, output_dim)
    policy = policy.to(device)
    policy.apply(init_weights)
    epsilon = start_epsilon

    if args.optim == 'adam':
        optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    else:
        assert args.optim == 'rmsprop'
        optimizer = optim.RMSprop(policy.parameters(), lr=args.lr)

    for episode in tqdm.tqdm(range(n_episodes), desc=f'Run: {run}'):

        loss, train_reward = train(train_env, policy, optimizer, discount_factor, epsilon, device)

        if episode < exploration_time:
            epsilon = epsilons[episode]

        test_reward = evaluate(test_env, policy, device)
        
        train_rewards[run][episode] = train_reward
        test_rewards[run][episode] = test_reward

torch.save(train_rewards, 'checkpoints/'+name+'_train.pt')
torch.save(train_rewards, 'checkpoints/'+name+'_test.pt')