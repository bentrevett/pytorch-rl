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
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--episodes', type=int, default=50)
parser.add_argument('--max_steps', type=int, default=100_000)
parser.add_argument('--n_steps', type=int, default=250)
parser.add_argument('--discount_factor', type=float, default=0.99)
args = parser.parse_args()

assert args.max_steps % args.n_steps == 0

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

def train_mc(env, actor, critic, actor_optimizer, critic_optimizer, discount_factor):
    
    actor.train()
    critic.train()

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

    returns = calculate_returns_mc(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values, actor_optimizer, critic_optimizer)

    return policy_loss, value_loss, episode_reward

def train_ns(env, actor, critic, actor_optimizer, critic_optimizer, n_steps, discount_factor):
    
    log_prob_actions = torch.zeros(n_steps)
    entropies = torch.zeros(n_steps)
    values = torch.zeros(n_steps)
    rewards = torch.zeros(n_steps)
    masks = torch.zeros(n_steps)

    state = env.state

    for step in range(n_steps):

        state = torch.FloatTensor(state).unsqueeze(0)
        
        action_preds = actor(state)
        value_pred = critic(state).squeeze(-1)

        action_probs = F.softmax(action_preds, dim = -1)
                
        dist = distributions.Categorical(action_probs)

        action = dist.sample()
        
        log_prob_action = dist.log_prob(action)
        
        entropy = dist.entropy()
        
        state, reward, done, _ = env.step(action.item())

        log_prob_actions[step] = log_prob_action
        entropies[step] = entropy
        values[step] = value_pred
        rewards[step] = reward
        masks[step] = 1 - done
    
        if done:
            state = env.reset()
    
    next_value = critic(torch.FloatTensor(state).unsqueeze(0)).squeeze(-1)
    returns = calculate_returns_ns(rewards, next_value, masks, discount_factor)
    advantages = calculate_advantages(returns, values)
    
    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values, actor_optimizer, critic_optimizer)

    return policy_loss, value_loss

def evaluate(env, actor, critic):

    actor.eval()
    critic.eval()

    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        action_preds = actor(state)

        action_probs = F.softmax(action_preds, dim = -1)

        dist = distributions.Categorical(action_probs)

        action = dist.sample() 

        state, reward, done, _ = env.step(action.item())

        episode_reward += reward

    return episode_reward

def calculate_returns_mc(rewards, discount_factor, normalize = False):

    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
        
    return returns

def calculate_returns_ns(rewards, next_value, masks, discount_factor, normalize = False):
    
    returns = torch.zeros_like(rewards)
    R = next_value.item()
    
    for i, (r, m) in enumerate(zip(reversed(rewards), reversed(masks))):
        R = r + R * discount_factor * m
        returns[i] = R

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns

def calculate_advantages(returns, values, normalize = False):
    
    advantages = returns - values
    
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages

def update_policy(advantages, log_prob_actions, returns, values, actor_optimizer, critic_optimizer):
    
    advantages = advantages.detach()
    returns = returns.detach()

    policy_loss = - (advantages * log_prob_actions).mean()
    
    value_loss = F.smooth_l1_loss(returns, values)

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    policy_loss.backward()
    value_loss.backward()
    
    actor_optimizer.step()
    critic_optimizer.step()
    
    return policy_loss.item(), value_loss.item()

experiment_rewards = np.zeros((len(seeds), args.max_steps // args.n_steps))

for seed in seeds:

    train_env = gym.make(args.env)
    test_env = gym.make(args.env)

    train_env.seed(seed)
    test_env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor = MLP(input_dim, args.hidden_dim, output_dim, args.dropout)
    critic = MLP(input_dim, args.hidden_dim, 1, args.dropout)

    actor.apply(init_weights)
    critic.apply(init_weights)

    actor_optimizer = optim.Adam(actor.parameters(), lr = args.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr = args.lr)

    episode_rewards = []

    print('*** begin mc pre-training')

    for episode in range(args.episodes):

        policy_loss, value_loss, _ = train_mc(train_env, actor, critic, actor_optimizer, critic_optimizer, args.discount_factor)

        episode_reward = evaluate(test_env, actor, critic)

        print('mc', episode, episode_reward, policy_loss, value_loss)

    print('*** begin ns training')

    _ = train_env.reset()

    for i in range((args.max_steps // args.n_steps)):

        policy_loss, value_loss = train_ns(train_env, actor, critic, actor_optimizer, critic_optimizer, args.n_steps, args.discount_factor)

        episode_reward = evaluate(test_env, actor, critic)

        print('ns', i, episode_reward, policy_loss, value_loss)

        experiment_rewards[seed][i] = episode_reward

os.makedirs('results', exist_ok=True)

np.savetxt(f'results/mc_a2cssqsqs.txt', experiment_rewards, fmt='%d')