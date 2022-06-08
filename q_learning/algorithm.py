import math

import numpy as np
import pettingzoo
import random
import time
import qchess_env

# TODO: enable multi agent training

env = qchess_env.env()

# ! math.prod needs python >= 3.8
action_space_size = math.prod(env.observation_spaces["player_0"].spaces["observation"].shape)
state_space_size = env.action_spaces["player_0"].n

# action_space_size = env.action_space_size
# state_space_size = env.state_space_size

q_table = np.zeros((state_space_size, action_space_size))

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

rewards_all_episodes = []
# Q-learning algorithm
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            try:
                action = np.random.choice(np.argwhere(env.observe(env.agent_selection)["action_mask"] == 1).reshape(-1))
            except:
                raise Exception("Can not perform any action.")
        # Take new action
        env.step(action)
        new_state, reward, done, info = env.last()
        # Update Q-table
        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (
            reward + discount_rate * np.max(q_table[new_state, :]))
        # Set new state
        state = new_state
        # Add new reward
        rewards_current_episode += reward
        if done == True:
            break
    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * \
        np.exp(-exploration_decay_rate*episode)
    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)

rewards_per_thousand_episodes = np.split(
    np.array(rewards_all_episodes), num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
