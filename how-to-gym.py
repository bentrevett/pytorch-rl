import torch
import gym

env = gym.make('Pendulum-v0')

print('action space is:', env.action_space)
print('action space shape:', env.action_space.shape)
print('is it continuous (box)?', isinstance(env.action_space, gym.spaces.Box))
print('high and low values?', env.action_space.high, env.action_space.low)
print('observation space is:', env.observation_space)
print('high and low values?', env.observation_space.high, env.observation_space.low)

env = gym.make('CartPole-v1')

print('action space is:', env.action_space)
print('action space shape', env.action_space.n) #notice it is .n and not n.shape for continuous
print('is it continuous (box)?', isinstance(env.action_space, gym.spaces.Box))
print('observation space is:', env.observation_space)
print('high and low values?', env.observation_space.high, env.observation_space.low)

for e in range(10):
    observation = env.reset()
    for t in range(1000):
        action = env.action_space.sample() #select random action
        observation, reward, done, info = env.step(action) #perform action on environment
        if done:
            print(f'Episode {e+1} finished after {t+1} timesteps')
            break
env.close()

