import os
from tqdm import tqdm
import numpy as np
import gym
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v1')
parser.add_argument('--n_seeds', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--n_layers', type=int, default=0)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--episodes', type=int, default=500)
parser.add_argument('--discount_factor', type=float, default=0.99)
parser.add_argument('--grad_clip', type=float, default=0.1)
parser.add_argument('--ppo_steps', type=int, default=5)
parser.add_argument('--ppo_clip', type=float, default=0.2)
args = parser.parse_args()

print(args)

env = gym.make(args.env)

assert isinstance(env.observation_space, gym.spaces.Box), 'Environment must have continuous inputs'
assert isinstance(env.action_space, gym.spaces.Discrete), 'Environment must have discrete outputs'

seeds = [s for s in range(args.n_seeds)]
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, activation, dropout):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fcs = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        activations = {'relu': F.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}
        self.activation = activations[activation]

    def forward(self, x):
        
        x = self.activation(self.dropout(self.fc_in(x)))

        for fc in self.fcs:
            x = self.activation(self.dropout(fc(x)))

        x = self.fc_out(x)

        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def train(env, actor, critic, actor_optimizer, critic_optimizer, discount_factor, ppo_steps, ppo_clip):

    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        states.append(state)

        action_preds = actor(state)
        value_pred = critic(state)

        action_probs = F.softmax(action_preds, dim = -1)

        dist = distributions.Categorical(action_probs)

        action = dist.sample()

        log_prob_action = dist.log_prob(action)
        
        state, reward, done, _ = env.step(action.item())

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)

        episode_reward += reward

    states = torch.cat(states)
    actions = torch.cat(actions)
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(actor, critic, states, actions, log_prob_actions, advantages, returns, actor_optimizer, critic_optimizer, ppo_steps, ppo_clip)

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

def update_policy(actor, critic, states, actions, log_prob_actions, advantages, returns, actor_optimizer, critic_optimizer, ppo_steps, ppo_clip):

    total_policy_loss = 0 
    total_value_loss = 0

    advantages = advantages.detach()
    returns = returns.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    
    for _ in range(ppo_steps):

        #get new log prob of actions for all input states
        action_pred = actor(states)
        values = critic(states).squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)

        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)

        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()

        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages

        policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

        value_loss = F.smooth_l1_loss(returns, values).mean()

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
        nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)

        actor_optimizer.step()
        critic_optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

experiment_rewards = np.zeros((len(seeds), args.episodes))

for seed in seeds:

    env = gym.make(args.env)

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor = MLP(input_dim, args.hidden_dim, output_dim, args.n_layers, args.activation, args.dropout)
    critic = MLP(input_dim, args.hidden_dim, 1, args.n_layers, args.activation, args.dropout)

    actor.apply(init_weights)
    critic.apply(init_weights)

    actor_optimizer = optim.Adam(actor.parameters(), lr = args.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr = args.lr)

    episode_rewards = []

    for episode in tqdm(range(args.episodes)):

        policy_loss, value_loss, episode_reward = train(env, actor, critic, actor_optimizer, critic_optimizer, args.discount_factor, args.ppo_steps, args.ppo_clip)

        episode_rewards.append(episode_reward)

        experiment_rewards[seed][episode] = episode_reward

os.makedirs('results', exist_ok=True)

np.savetxt(f'results/mc_ppo_{args.hidden_dim}hd_{args.n_layers}nl_{args.activation}ac_{args.dropout}do_{args.lr}lr_{args.discount_factor}df_{args.grad_clip}gc_{args.ppo_steps}ps_{args.ppo_clip}pc.txt', experiment_rewards, fmt='%d')