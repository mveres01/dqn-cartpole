"""
Script adapted from:
https://github.com/yukezhu/tensorflow-reinforce/blob/master/run_dqn_cartpole.py
"""

from __future__ import print_function
from collections import deque

import numpy as np
import gym
import dqn

env_name = 'CartPole-v0'
env = gym.make(env_name)

state_shape = env.observation_space.shape
num_actions = env.action_space.n
batch_size = 64
q_learner = dqn.Agent(state_shape, num_actions, batch_size=batch_size)

MAX_EPISODES = 10000
MAX_STEPS = 200

episode_history = deque(maxlen=100)
for i in xrange(MAX_EPISODES):

  # initialize
  state = env.reset()
  total_rewards = 0

  for t in range(MAX_STEPS):
    env.render()
    action = q_learner.choose_action(state)

    next_state, reward, done, _ = env.step(action)

    total_rewards += reward

    q_learner.update_buffer(state, action, reward, next_state, done)

    # Only start learning after buffer has some experience in it
    if i > 50:
        q_learner.update_policy()


    state = next_state
    if done: 
        break

  episode_history.append(total_rewards)
  mean_rewards = np.mean(episode_history)

  print("Episode {}".format(i))
  print("Finished after {} timesteps".format(t+1))
  print("Reward for this episode: {}".format(total_rewards))
  print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))

  if mean_rewards >= 195.0:
    print("Environment {} solved after {} episodes".format(env_name, i + 1))
    break
