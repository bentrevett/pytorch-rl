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
parser.add_argument('--hidden_dim', type=int, required=True)
args = parser.parse_args()

env = gym.make('CartPole-v1')

assert isinstance(env.observation_space, gym.spaces.Box)
assert isinstance(env.action_space, gym.spaces.Discrete)

N_SEEDS = 10
SEEDS = [s for s in range(N_SEEDS)]
INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = args.hidden_dim
print(HIDDEN_DIM)
OUTPUT_DIM = env.action_space.n
LEARNING_RATE = 0.01
MAX_EPISODES = 500
DISCOUNT_FACTOR = 0.99

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.0):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_a = nn.Linear(hidden_dim, output_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        return self.fc_a(x), self.fc_v(x)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def train(env, model, optimizer, discount_factor):
    
    log_prob_actions = []
    values = []
    rewards = []
    entropies = []
    masks = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        action_preds, value_pred = model(state)
        
        action_probs = F.softmax(action_preds, dim = -1)
                
        dist = distributions.Categorical(action_probs)

        action = dist.sample()
        
        log_prob_action = dist.log_prob(action)
        
        entropy = dist.entropy()

        state, reward, done, _ = env.step(action.item())

        log_prob_actions.append(log_prob_action)
        entropies.append(entropy)
        values.append(value_pred)
        rewards.append(reward)
        masks.append(1-done)

        episode_reward += reward

    log_prob_actions = torch.cat(log_prob_actions)
    entropies = torch.cat(entropies)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, masks, discount_factor)
        
    policy_loss, value_loss = update_policy(returns, log_prob_actions, values, entropies, optimizer)

    return policy_loss, value_loss, episode_reward

def calculate_returns(rewards, masks, discount_factor, normalize = True):
    
    returns = []
    R = 0
    
    for r, m in zip(reversed(rewards), reversed(masks)):
        R = r + R * discount_factor * m
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    
    if normalize:
        
        returns = (returns - returns.mean()) / returns.std()
        
    return returns

def update_policy(returns, log_prob_actions, values, entropies, optimizer):
    
    returns = returns.detach()

    policy_loss = - (returns * log_prob_actions).mean()
    
    value_loss = 0.5 * F.smooth_l1_loss(returns, values).mean()
    
    loss = policy_loss + value_loss - 0.001 * entropies.mean()

    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    return policy_loss.item(), value_loss.item()

experiment_rewards = np.zeros((N_SEEDS, MAX_EPISODES))

for seed in SEEDS:

    env = gym.make('CartPole-v1')

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = MLP(INPUT_DIM, args.hidden_dim, OUTPUT_DIM)

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    episode_rewards = []

    for episode in tqdm(range(MAX_EPISODES)):

        policy_loss, value_loss, episode_reward = train(env, model, optimizer, DISCOUNT_FACTOR)

        episode_rewards.append(episode_reward)

        experiment_rewards[seed][episode] = episode_reward

os.makedirs('results', exist_ok=True)

np.savetxt(f'results/ac_single_valcoeff_ent_{args.hidden_dim}.txt', experiment_rewards, fmt='%d')