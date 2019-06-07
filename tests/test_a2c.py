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
parser.add_argument('--n_seeds', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--episodes', type=int, default=500)
parser.add_argument('--discount_factor', type=float, default=0.99)
args = parser.parse_args()

env = gym.make(args.env)

assert isinstance(env.observation_space, gym.spaces.Box)
assert isinstance(env.action_space, gym.spaces.Discrete)

seeds = [s for s in range(args.n_seeds)]
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def train(env, actor, critic, actor_optimizer, critic_optimizer, discount_factor):
    
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        action_preds = actor(state)
        value_pred = critic(state)
        
        action_probs = F.softmax(action_preds, dim = -1)
                
        dist = distributions.Categorical(action_probs)

        action = dist.sample()
        
        log_prob_action = dist.log_prob(action)
        
        state, reward, done, _ = env.step(action.item())

        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)

        episode_reward += reward

    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values, actor_optimizer, critic_optimizer)

    return policy_loss, value_loss, episode_reward

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

def calculate_advantages(returns, values, normalize = True):
    
    advantages = returns - values
    
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages

def update_policy(advantages, log_prob_actions, returns, values, actor_optimizer, critic_optimizer):
    
    advantages = advantages.detach()
    returns = returns.detach()

    policy_loss = - (advantages * log_prob_actions).mean()
    
    value_loss = F.smooth_l1_loss(returns, values).mean()
    
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    policy_loss.backward()
    value_loss.backward()
    
    actor_optimizer.step()
    critic_optimizer.step()
    
    return policy_loss.item(), value_loss.item()

experiment_rewards = np.zeros((len(seeds), args.episodes))

for seed in seeds:

    env = gym.make(args.env)

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor = MLP(input_dim, args.hidden_dim, output_dim, args.dropout)
    critic = MLP(input_dim, args.hidden_dim, 1, args.dropout)

    actor.apply(init_weights)
    critic.apply(init_weights)

    actor_optimizer = optim.Adam(actor.parameters(), lr = args.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr = args.lr)

    episode_rewards = []

    for episode in tqdm(range(args.episodes)):

        policy_loss, value_loss, episode_reward = train(env, actor, critic, actor_optimizer, critic_optimizer, args.discount_factor)

        episode_rewards.append(episode_reward)

        experiment_rewards[seed][episode] = episode_reward

os.makedirs('results', exist_ok=True)

np.savetxt('results/a2c.txt', experiment_rewards, fmt='%d')