import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import log_softmax, softmax, mse_loss, normalize
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_value_
from collections import deque


ALPHA = 0.0005              # learning rate for the actor
BETA = 0.0005               # learning rate for the critic
GAMMA = 0.99                # discount rate
HIDDEN_SIZE = 256           # number of hidden nodes we have in our approximation
PSI = 0.1                   # the entropy bonus multiplier

BATCH_SIZE = 25             # number of episodes in a batch
NUM_EPOCHS = 5000
NUM_STEPS = 7               # number of steps to bootstrap after

RENDER_EVERY = 100


# Q-table is replaced by a neural network
class Actor(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)
        )

    def forward(self, x):
        x = normalize(x, dim=1)
        x = self.net(x)
        return x


class Critic(nn.Module):
    def __init__(self, observation_space_size: int, hidden_size: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=1, bias=True)
        )

    def forward(self, x):
        x = normalize(x, dim=1)
        x = self.net(x)
        return x


def get_discounted_returns(rewards: torch.Tensor, gamma: float, state_values: torch.Tensor, n: int):
    """
        Computes the array of discounted rewards [Gt:t+1] for the episode. See reference on p.143 S&B.
        Args:
            rewards: the sequence of the rewards obtained from running the episode
            gamma: the discounting factor
            state_values: teh values of the states calculated by the critic network
            n: the horizon of the bootstrapping
        Returns:
            discounted_rewards: the sequence of the discounted returns from time step t
    """
    discounted_rewards = torch.empty_like(rewards)
    gamma_array = torch.full(size=(n+1,), fill_value=gamma) if n != 1 else None
    power_gamma_array = torch.pow(gamma_array, torch.arange(n+1).float()) if n != 1 else None

    # # turn the state values torch tensor into the numpy array
    # state_values = state_values.numpy()

    # define the end of sequence
    T = rewards.shape[0]

    # for every time step in the sequence
    for t in range(T):

        # special case of 1 step lookahead bootstrapping
        if n == 1:

            # check if we can discount
            if t < T - 1:
                Gt = rewards[t] + gamma * state_values[t+1]

            else:
                # the last reward
                Gt = rewards[T-1]

        # check if we can bootstrap
        elif t + n < T:
            # calculate the bootstrapped return
            Gt = torch.sum(power_gamma_array[:-1] * rewards[t:(t+n)]) + power_gamma_array[-1] * state_values[t+n]

        # if we can't bootstrap anymore
        else:

            # check if we can discount
            if t < T - 1:
                # compute the monte carlo return
                Gt = torch.sum(power_gamma_array[:rewards[t:T].shape[0]] * rewards[t:T])

            else:
                # the last reward
                Gt = rewards[T-1]

        discounted_rewards[t] = Gt

    return discounted_rewards


def get_entropy_bonus(logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
        Calculates the entropy bonus.
        Args:
            logits: the logits of the actor network
        Returns:
            entropy_bonus: entropy bonus
            mean_entropy: the mean entropy of the episode
    """
    # calculate the probabilities
    p = softmax(logits, dim=1)

    # calculate the log probabilities
    log_p = log_softmax(logits, dim=1)

    # calculate the entropy
    entropy = -1 * torch.sum(p * log_p, dim=1)

    # calculate the mean entropy for the episode
    mean_entropy = torch.mean(entropy, dim=0)

    # calculate the entropy bonus
    entropy_bonus = -1 * PSI * mean_entropy

    return entropy_bonus, mean_entropy


def play_episode(env: gym.Env, actor: nn.Module, critic: nn.Module, epoch: int, episode: int):
    """
        Plays an episode of the environment.
        Args:
            env: the OpenAI environment
            actor: the policy network
            critic: the state value function
            epoch: current epoch
            episode: current episode
        Returns:
            state_values: the values of the states as calculated by the critic network
            action_log_probs: log-probabilities of the takes actions in the trajectory
            rewards: the sequence of the obtained rewards
            logits: the logits of every action taken - needed to compute entropy for entropy bonus
            episode_total_reward: sum of the rewards for the episode - needed for the average over 200 episode statistic
    """
    # initialize the environment state
    current_state = env.reset()

    logits = torch.empty(size=(0, env.action_space.n), dtype=torch.float)
    action_log_probs = torch.empty(size=(0,), dtype=torch.float)
    state_values = torch.empty(size=(0,), dtype=torch.float)
    rewards = torch.empty(size=(0,), dtype=torch.float)

    # set the done flag to false
    done = False

    # init the total reward
    episode_total_reward = 0

    # accumulate data for 1 episode
    while not done:

        # render the episode
        if epoch % RENDER_EVERY == 0 and episode == 0:
            env.render()

        # get the action logits from the agent - (preferences)
        action_logits = actor(torch.tensor(current_state).float().unsqueeze(dim=0)).squeeze()

        # append the logits
        logits = torch.cat((logits, action_logits.unsqueeze(dim=0)), dim=0)

        # sample an action according to the action distribution
        action = Categorical(logits=action_logits).sample()

        # compute the log-probabilities of the actions
        log_probs = log_softmax(action_logits, dim=0)

        # get the log-probability of the chosen action
        action_log_probs = torch.cat((action_log_probs, log_probs[action.item()].unsqueeze(dim=0)), dim=0)

        # get the current state value
        current_state_value = critic(torch.tensor(current_state).float().unsqueeze(dim=0))
        state_values = torch.cat((state_values, current_state_value), dim=0)

        # take the action
        new_state, reward, done, _ = env.step(action.item())

        episode_total_reward += reward

        # save the reward
        rewards = torch.cat((rewards, torch.tensor(reward, dtype=torch.float).unsqueeze(dim=0)), dim=0)

        # if the episode is over
        if done:
            break

        # update the state
        current_state = new_state

    return state_values, action_log_probs, rewards, logits, episode_total_reward


def main():

    # create the environment
    env = gym.make('LunarLander-v2')

    # policy network
    actor = Actor(observation_space_size=env.observation_space.shape[0],
                  action_space_size=env.action_space.n,
                  hidden_size=HIDDEN_SIZE)

    # state-value network
    critic = Critic(observation_space_size=env.observation_space.shape[0],
                    hidden_size=HIDDEN_SIZE)

    # define the optimizers for the policy and state-value networks
    adam_actor = optim.Adam(params=actor.parameters(), lr=ALPHA)
    adam_critic = optim.Adam(params=critic.parameters(), lr=BETA)

    total_rewards = deque([], maxlen=100)

    # run for N epochs
    for epoch in range(NUM_EPOCHS):

        # holder for the weighted log-probs
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float)

        # holder for the epoch logits
        epoch_logits = torch.empty(size=(0, env.action_space.n), dtype=torch.float)

        # holder for the epoch state values
        epoch_state_values = torch.empty(size=(0,), dtype=torch.float)

        # holder for the epoch discounted returns
        epoch_discounted_returns = torch.empty(size=(0,), dtype=torch.float)

        # collect the data from the episode
        for episode in range(BATCH_SIZE):

            # play an episode
            (state_values,
             action_log_probs,
             rewards,
             logits,
             episode_total_reward) = play_episode(env=env, actor=actor, critic=critic, epoch=epoch, episode=episode)

            # calculate the sequence of the discounted returns Gt
            discounted_returns = get_discounted_returns(rewards=rewards,
                                                        gamma=GAMMA,
                                                        state_values=state_values.detach().squeeze(),
                                                        n=NUM_STEPS)

            # calculate the advantage for time t: Q(s,a) - V(s)
            advantages = discounted_returns - state_values.detach().squeeze()

            # append sum of logP * A
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs,
                                                  torch.sum(action_log_probs * advantages).unsqueeze(dim=0)), dim=0)

            # append the logits for the entropy bonus
            epoch_logits = torch.cat((epoch_logits, logits), dim=0)

            # append the state values
            epoch_state_values = torch.cat((epoch_state_values, state_values), dim=0)

            # append the discounted returns
            epoch_discounted_returns = torch.cat((epoch_discounted_returns, discounted_returns), dim=0)

            # append the episodic total rewards
            total_rewards.append(episode_total_reward)

        # calculate the policy loss
        policy_loss = -1 * torch.mean(epoch_weighted_log_probs)

        # get the entropy bonus
        entropy_bonus, mean_entropy = get_entropy_bonus(logits=epoch_logits)

        # add the entropy bonus
        policy_loss += (PSI * entropy_bonus)

        # zero the gradient in both actor and the critic networks
        actor.zero_grad()
        critic.zero_grad()

        # calculate the policy gradient
        policy_loss.backward()

        # calculate the critic loss
        critic_loss = mse_loss(input=epoch_state_values.squeeze(), target=epoch_discounted_returns)

        # calculate the gradient of the critic loss
        critic_loss.backward()

        # clip the gradients in the policy gradients and the critic loss gradients
        clip_grad_value_(parameters=actor.parameters(), clip_value=0.1)
        clip_grad_value_(parameters=critic.parameters(), clip_value=0.1)

        # update the actor and critic parameters
        adam_actor.step()
        adam_critic.step()

        print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(total_rewards):.3f}", end="", flush=True)

        # check if solved
        if np.mean(total_rewards) > 200:
            print('\nSolved!')
            break

    # close the environment
    env.close()

if __name__ == "__main__":
    main()