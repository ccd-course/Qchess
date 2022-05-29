import numpy as np
import gym
import torch
import torch.nn.functional as F
import time
from itertools import count
import math
from network import ADRQN
from experience_replay import ExpBuffer
from env import EnvManager
from agent import Agent
from epsilon_greedy_strategy import EpsilonGreedyStrategy
from plot import plot, get_moving_average

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = EnvManager(device, 'CartPole-v1')

state_size = env.num_state_features()
n_actions = env.num_actions_available()

embedding_size = 8
M_episodes = 2500
memory_size = 10000
sample_length = 20
replay_buffer = ExpBuffer(memory_size, sample_length)
batch_size = 256
eps_start = 0.9
eps = eps_start
eps_end = 0.05
eps_decay = 10
gamma = 0.999
learning_rate = 0.01
blind_prob = 0
EXPLORE = 300
target_update = 5  # Interval for updating the target net

strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, env.num_actions_available(), device)
adrqn = ADRQN(n_actions, state_size, embedding_size).cuda()
adrqn_target = ADRQN(n_actions, state_size, embedding_size).cuda()
adrqn_target.load_state_dict(adrqn.state_dict())

optimizer = torch.optim.Adam(adrqn.parameters(), lr=learning_rate)

episode_durations = []

for episode in range(M_episodes):
    done = False
    hidden = None
    last_action = 0
    current_return = 0
    env.reset()
    last_observation = env.get_state()
    for timestep in count():
        action, hidden = agent.act(torch.tensor(last_observation).float().view(1, 1, -1).cuda(), F.one_hot(
            torch.tensor(last_action), n_actions).view(1, 1, -1).float().cuda(), eps, adrqn, hidden=hidden)

        observation, reward, done, info = env.take_action(action)
        if np.random.rand() < blind_prob:
            # Induce partial observability
            observation = np.zeros_like(observation)

        reward = np.sign(reward)
        current_return += reward
        replay_buffer.write_tuple(
            (last_action, last_observation, action, reward, observation, done))

        last_action = action
        last_observation = observation

        # Updating Networks
        if episode > EXPLORE:
            eps = eps_end + (eps_start - eps_end) * \
                math.exp((-1*(episode-EXPLORE))/eps_decay)

            last_actions, last_observations, actions, rewards, observations, dones = replay_buffer.sample(
                batch_size)
            q_values, _ = adrqn.forward(
                last_observations, F.one_hot(last_actions, n_actions).float())
            q_values = torch.gather(
                q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
            predicted_q_values, _ = adrqn_target.forward(
                observations, F.one_hot(actions, n_actions).float())
            target_values = rewards + \
                (gamma * (1 - dones.float()) *
                 torch.max(predicted_q_values, dim=-1)[0])

            # Update network parameters
            optimizer.zero_grad()
            loss = torch.nn.MSELoss()(q_values, target_values.detach())
            loss.backward()
            optimizer.step()

        if done:
            episode_durations.append(timestep)
            plot(episode_durations, 10)
            break

    if episode % target_update == 0:
        adrqn_target.load_state_dict(adrqn.state_dict())

    if get_moving_average(10, episode_durations)[-1] >= 195:
        break

env.close()
