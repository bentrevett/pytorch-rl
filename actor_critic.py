import collections

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
hidden_dim = 32
n_episodes = 2500
batch_size = 32
gamma = 0.99
decay = 0.97

env = gym.make('CartPole-v1')

env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, output_dim)
                      )

    def forward(self, state):
        return self.fc(state)

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_preds = self.actor(state)
        value_pred = self.critic(state)
        return action_preds, value_pred

class Transition:
    def __init__(self, state, action, reward, done, log_prob_action, value):
        
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.log_prob_action = log_prob_action
        self.value = value

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

actor = MLP(input_dim, hidden_dim, output_dim)
critic = MLP(input_dim, hidden_dim, 1)
model = ActorCritic(actor, critic)

actor_optimizer = optim.Adam(model.actor.parameters())
critic_optimizer = optim.Adam(model.critic.parameters())

transitions = []
episode_rewards = []

for episode in tqdm(range(n_episodes)):

    episode_reward = 0

    state = env.reset()

    while True:

        state = torch.FloatTensor(state).unsqueeze(0)

        action_preds, value_pred = model(state)

        action_probs = F.softmax(action_preds, dim = -1)

        dist = distributions.Categorical(action_probs)

        action = dist.sample()

        log_prob_action = dist.log_prob(action)

        next_state, reward, done, _ = env.step(action.item())

        transitions.append(Transition(state, 
                                      action, 
                                      torch.FloatTensor([reward]), 
                                      torch.FloatTensor([done]),
                                      log_prob_action,
                                      value_pred 
                                     )
                          )
            
        episode_reward += reward

        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            break

    if len(transitions) >= batch_size:

        with torch.no_grad():

            reward_to_go = torch.FloatTensor([0])
            advantage = torch.FloatTensor([0])
            next_value = torch.FloatTensor([0])

            for transition in reversed(transitions):

                reward_to_go = transition.reward + (1 - transition.done) * gamma * reward_to_go
                transition.reward_to_go = reward_to_go
                td_error = transition.reward + (1 - transition.done) * gamma * next_value - transition.value
                advantage = td_error + (1 - transition.done) * gamma * decay * advantage
                transition.advantage = advantage
                next_value = transition.value

        advantage = torch.cat([t.advantage for t in transitions])

        #advantage = (advantage - advantage.mean()) / advantage.std()

        log_prob_action = torch.cat([t.log_prob_action for t in transitions])

        policy_loss = -(log_prob_action.sum() * advantage).mean()
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

        value = torch.cat([t.value for t in transitions])
        reward_to_go = torch.cat([t.reward_to_go for t in transitions])

        value_loss = (value - reward_to_go).pow(2).mean()
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()

        transitions = []

fig = plt.figure()
plt.plot(episode_rewards)
fig.savefig('rewards')
    


    