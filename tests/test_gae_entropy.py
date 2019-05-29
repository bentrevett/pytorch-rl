import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

from tqdm import tqdm
import numpy as np
import gym

env = gym.make('CartPole-v1')

assert isinstance(env.observation_space, gym.spaces.Box)
assert isinstance(env.action_space, gym.spaces.Discrete)

N_SEEDS = 10
SEEDS = [s for s in range(N_SEEDS)]
INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = env.action_space.n
LEARNING_RATE = 0.01
MAX_EPISODES = 500
DISCOUNT_FACTOR = 0.99
ENTROPY_WEIGHT = 0.01
TRACE_DECAY = 0.95

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.25):
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

def train(env, actor, critic, actor_optimizer, critic_optimizer, discount_factor, entropy_weight, trace_decay):
    
    log_prob_actions = []
    entropies = []
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

        entropy = dist.entropy()

        action = dist.sample()
        
        log_prob_action = dist.log_prob(action)
        
        state, reward, done, _ = env.step(action.item())

        log_prob_actions.append(log_prob_action)
        entropies.append(entropy)
        values.append(value_pred)
        rewards.append(reward)

        episode_reward += reward

    log_prob_actions = torch.cat(log_prob_actions)
    entropies = torch.cat(entropies)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(rewards, values, discount_factor, trace_decay)
        
    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values, entropies, actor_optimizer, critic_optimizer, entropy_weight)

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

def calculate_advantages(rewards, values, discount_factor, trace_decay, normalize = True):
    
    advantages = []
    advantage = 0
    next_value = 0
    
    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + next_value * discount_factor - v
        advantage = td_error + advantage * discount_factor * trace_decay
        next_value = v
        advantages.insert(0, advantage)
        
    advantages = torch.tensor(advantages)
    
    if normalize:
        
        advantages = (advantages - advantages.mean()) / advantages.std()
        
    return advantages

def update_policy(advantages, log_prob_actions, returns, values, entropies, actor_optimizer, critic_optimizer, entropy_weight):
    
    policy_loss = - (advantages * log_prob_actions).mean() - entropy_weight * entropies.mean()
    
    value_loss = F.smooth_l1_loss(returns, values).mean()
    
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    policy_loss.backward()
    value_loss.backward()
    
    actor_optimizer.step()
    critic_optimizer.step()
    
    return policy_loss.item(), value_loss.item()

experiment_rewards = np.zeros((N_SEEDS, MAX_EPISODES))

for seed in SEEDS:

    env = gym.make('CartPole-v1')

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    actor.apply(init_weights)
    critic.apply(init_weights)

    actor_optimizer = optim.Adam(actor.parameters(), lr = LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr = LEARNING_RATE)

    episode_rewards = []

    for episode in tqdm(range(MAX_EPISODES)):

        policy_loss, value_loss, episode_reward = train(env, actor, critic, actor_optimizer, critic_optimizer, DISCOUNT_FACTOR, ENTROPY_WEIGHT, TRACE_DECAY)

        episode_rewards.append(episode_reward)

        experiment_rewards[seed][episode] = episode_reward

os.makedirs('results', exist_ok=True)

np.savetxt('results/gae_entropy.txt', experiment_rewards, fmt='%d')