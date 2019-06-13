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

from utils import SubprocVecEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v1')
parser.add_argument('--n_seeds', type=int, default=5)
parser.add_argument('--n_envs', type=int, default=16)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--n_layers', type=int, default=0)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--discount_factor', type=float, default=0.99)
parser.add_argument('--grad_clip', type=float, default=0.1)
parser.add_argument('--n_steps', type=int, default=100)
parser.add_argument('--max_steps', type=int, default=100_000)
parser.add_argument('--n_evaluations', type=int, default=3)
args = parser.parse_args()

print(args)

assert args.n_seeds > 0
assert args.n_envs > 0
assert args.hidden_dim > 0
assert args.n_layers >= 0
assert args.activation in ['relu', 'tanh', 'sigmoid']
assert args.dropout >= 0 and args.dropout < 1.0
assert args.lr > 0
assert args.episodes >= 0
assert args.max_steps >= 0
assert args.n_steps >= 0
assert args.discount_factor >= 0 and args.discount_factor <= 1
assert args.grad_clip > 0
assert args.n_evaluations > 0

#run "experiment" on seed = 1 to seed = n_seeds
seeds = [s for s in range(0, args.n_seeds)]

#make sure environment is a valid gym environment
env = gym.make(args.env)

#check environment has continuous input and discrete output
assert isinstance(env.observation_space, gym.spaces.Box)
assert isinstance(env.action_space, gym.spaces.Discrete)

#get input and output dimensions for model/policy
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

#delete environment afterwards
del env
 
def make_env(env_name, seed):
    """
    Used by SubprocVecEnc to create multiple environments
    """
    def _thunk():
        env = gym.make(env_name)
        env.seed(seed)
        return env

    return _thunk

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, activation, dropout):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fcs = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        self.fc_in = nn.Linear(hidden_dim, output_dim)
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
    """
    Initialize weights by the Xavier normal initialization scheme
    Initialze biases to zero
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def train(envs, actor, critic, actor_optimizer, critic_optimizer, n_steps, discount_factor):
    """
    Train using the n-step setting
    Interact with environment for n_steps
    Use rewards to calculate returns
    Use returns to calculate advantage (return - predicted value)
    Update policy
    """

    #tensors to store values used for updating policy
    log_prob_actions = torch.zeros(n_steps, len(envs))
    values = torch.zeros(n_steps, len(envs))
    rewards = torch.zeros(n_steps, len(envs))
    masks = torch.zeros(n_steps, len(envs)) #mask is 1 when state is non-terminal, 0 for terminal states
    entropies = torch.zeros(n_steps, len(envs))

    #get state (numpy array) for all environments
    state = envs.get_state() #[n_envs, observation_space]

    assert state.shape == (args.n_envs, input_dim)

    for step in range(n_steps):

        #convert state into tensor
        state = torch.FloatTensor(state) #[n_envs, observation_space]
        
        assert state.shape == (args.n_envs, input_dim)

        #action logits from actor
        #value preds from critic
        action_preds = actor(state) #[n_envs, action_space]
        value_pred = critic(state).squeeze(-1) #[n_envs]

        assert action_preds.shape == (args.n_envs, output_dim)
        assert value_pred.shape == (args.n_envs,)

        #convert logits to probabilities
        action_probs = F.softmax(action_preds, dim = -1) #[n_envs, action_space]

        #create probability distribution
        dist = distributions.Categorical(action_probs)

        #sample probability distribution for each environment
        action = dist.sample() #[n_envs]

        assert action.shape == (args.n_envs,)

        #calculate log probability of each action sampled
        log_prob_action = dist.log_prob(action) #[n_envs]

        assert log_prob_action.shape == (args.n_envs,)

        #get entropy for debugging
        entropy = dist.entropy()

        assert entropy.shape == (args.n_envs,)

        #action now numpy array across all envs
        #state = [n_envs, observation space]
        #reward = [n_envs]
        #done = [n_envs]
        state, reward, done, _ = envs.step(action.cpu().numpy())

        assert state.shape == (args.n_envs, input_dim)
        assert reward.shape == (args.n_envs,)
        assert done.shape == (args.n_envs,)

        #convert to tensor and create mask from 'done'
        reward = torch.FloatTensor(reward) #[n_envs]
        mask = torch.FloatTensor(1-done) #[n_envs]

        #store each in tensor
        log_prob_actions[step] = log_prob_action
        values[step] = value_pred
        rewards[step] = reward
        masks[step] = mask
        entropies[step] = entropy

    #return from step n+1 taken from critic of following state
    next_value = critic(torch.FloatTensor(state)).squeeze(-1)

    assert next_value.shape == (args.n_envs,)

    #calculate returns and advantages
    returns = calculate_returns(rewards, next_value, masks, discount_factor)
    advantages = calculate_advantages(returns, values)
    
    #update policy
    policy_loss, value_loss = update_policy(actor, critic, advantages, log_prob_actions, returns, values, actor_optimizer, critic_optimizer)

    return policy_loss, value_loss, entropies.mean()

def calculate_returns(rewards, next_value, masks, discount_factor, normalize = False):
    """
    Calculates returns (discounted rewards) for n-step updates
    next_value is predicted value from critic from n+1 state 
    """

    assert rewards.shape == masks.shape == (args.n_steps, args.n_envs)
    assert next_value.shape == (args.n_envs,)

    returns = torch.zeros_like(rewards) #[n_steps, n_envs]
    R = next_value
    
    for i, (r, m) in enumerate(zip(reversed(rewards), reversed(masks))):
        R = r + R * discount_factor * m
        returns[i] = R

    assert i == (args.n_steps - 1)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns

def calculate_advantages(returns, values, normalize = True):
    """
    Calculate advantages (returns - values)
    """

    assert returns.shape == values.shape

    #for Monte-Carlo, values and advantages are shape [length of episode]
    #for n-step, all three are [n_steps, n_envs]
    advantages = returns - values 

    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages

def update_policy(actor, critic, advantages, log_prob_actions, returns, values, actor_optimizer, critic_optimizer):
    """
    Update actor and critic
    Calculate policy loss and average over all environments and steps
    Calculate value loss and average over all environments and steps
    Calculate gradients
    Update actor and critic parameters
    """

    #detach as don't want to backpropagate through these
    advantages = advantages.detach()
    returns = returns.detach()

    # - adv * log(p(a|s)), averaged over all environments and steps
    policy_loss = - (advantages * log_prob_actions).mean()

    # l1 loss between return and value
    value_loss = F.smooth_l1_loss(returns, values).mean()

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    #calculate gradients
    policy_loss.backward()
    value_loss.backward()
    
    #clip gradients
    nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
    nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)

    #update parameters
    actor_optimizer.step()
    critic_optimizer.step()
    
    return policy_loss.item(), value_loss.item()

def evaluate(env, actor, critic):
    """
    Run policy on environment for single episode, return cumulative reward
    """

    #turn off dropout/batchnorm
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

############################
# MAIN ENTRY POINT IS HERE #
############################

#numpy array for storing results
#for each seed contains the cumulative reward evaluated after every n_steps
experiment_rewards = np.zeros((len(seeds), args.max_steps // args.n_steps))

#run experiments n_seeds times
for seed in seeds:

    #seed numpy and torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    #create environments for training
    #ensure all have different seeds across each base seed
    train_envs = [make_env(args.env, (seed * args.n_envs)+i) for i in range(args.n_envs)]
    train_envs = SubprocVecEnv(train_envs)

    #environment we evaluate on has different seed from all environment we train on
    test_env = gym.make(args.env)
    test_env.seed(seed+args.n_envs)

    #create actor and critic
    actor = MLP(input_dim, args.hidden_dim, output_dim, args.n_layers, args.activation, args.dropout)
    critic = MLP(input_dim, args.hidden_dim, 1, args.n_layers, args.activation, args.dropout)

    #initialize weights for actor and critic
    actor.apply(init_weights)
    critic.apply(init_weights)

    #initialize optimizer for actor and critic
    actor_optimizer = optim.Adam(actor.parameters(), lr = args.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr = args.lr)

    _ = train_envs.reset()

    for i in range((args.max_steps // args.n_steps)):

        policy_loss, value_loss, entropy = train(train_envs, actor, critic, actor_optimizer, critic_optimizer, args.n_steps, args.discount_factor)

        episode_reward = np.mean([evaluate(test_env, actor, critic) for _ in range(args.n_evaluations)])

        print(f'Updates: {i+1:4}, Steps: {(i+1)*args.n_steps:6}, Reward: {episode_reward:5.1f}, Entropy: {entropy:.3f}, Val. Loss: {value_loss:5.2f}')

        experiment_rewards[seed][i] = episode_reward

os.makedirs('results', exist_ok=True)

np.savetxt(f'results/ns_a2c_{args.hidden_dim}hd_{args.n_layers}nl_{args.activation}ac_{args.dropout}do_{args.lr}lr_{args.discount_factor}df_{args.grad_clip}gc_{args.n_steps}ns.txt', experiment_rewards, fmt='%d')