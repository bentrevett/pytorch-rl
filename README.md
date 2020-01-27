# PyTorch Reinforcement Learning

This repo contains tutorials covering reinforcement learning using PyTorch 1.3 and Gym 0.15.4 using Python 3.7.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-rl/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

To install Gym, see installation instructions on the [Gym GitHub repo](https://github.com/openai/gym).

## Tutorials

All tutorials use Monte Carlo methods to train the CartPole-v1 environment with the goal of reaching a total episode reward of 475 averaged over the last 25 episodes. There are also alternate versions of some algorithms to show how to use those algorithms with other environments.

* 0 - [Introduction to Gym](https://github.com/bentrevett/pytorch-rl/blob/master/0%20-%20Introduction%20to%20Gym.ipynb)

* 1 - [Vanilla Policy Gradient (REINFORCE)](https://github.com/bentrevett/pytorch-rl/blob/master/1%20-%20Vanilla%20Policy%20Gradient%20(REINFORCE)%20[CartPole].ipynb)

    This tutorial covers the workflow of a reinforcement learning project. We'll learn how to: create an environment, initialize a model to act as our policy, create a state/action/reward loop and update our policy. We update our policy with the [vanilla policy gradient algorithm](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), also known as [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf).
    
* 2 - [Actor Critic](https://github.com/bentrevett/pytorch-rl/blob/master/2%20-%20Actor%20Critic.ipynb)

    This tutorial introduces the family of [actor-critic](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf) algorithms, which we will use for the next few tutorials.

* 3 - [Advantage Actor Critic (A2C)](https://github.com/bentrevett/pytorch-rl/blob/master/3%20-%20Advantage%20Actor%20Critic%20(A2C)%20[CartPole].ipynb)

    We cover an improvement to the actor-critic framework, the [A2C](https://arxiv.org/abs/1602.01783) (advantage actor-critic) algorithm.
    
* 4 - [Generalized Advantage Estimation (GAE)](https://github.com/bentrevett/pytorch-rl/blob/master/4%20-%20Generalized%20Advantage%20Estimation%20(GAE)%20[CartPole].ipynb)

    We improve on A2C by adding [GAE](https://arxiv.org/abs/1506.02438) (generalized advantage estimation). 

* 5 - [Proximal Policy Evaluation](https://github.com/bentrevett/pytorch-rl/blob/master/5%20-%20Proximal%20Policy%20Optimization%20(PPO)%20[CartPole].ipynb)

    We cover another improvement on A2C, [PPO](https://arxiv.org/abs/1707.06347) (proximal policy optimization).

Potential algorithms covered in future tutorials: DQN, ACER, ACKTR.

## References

* 'Reinforcement Learning: An Introduction' - http://incompleteideas.net/sutton/book/the-book-2nd.html
* 'Algorithms for Reinforcement Learning' - https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf
* List of key papers in deep reinforcement learning - https://spinningup.openai.com/en/latest/spinningup/keypapers.html