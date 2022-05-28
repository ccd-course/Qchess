import numpy as np
import gym
import torch
import torch.nn.functional as F
import time
from itertools import count
import math
from network import ADRQN
from experience_replay import ExpBuffer


env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
n_actions = env.action_space.n
embedding_size = 8
M_episodes = 2500
replay_buffer_size = 100000
sample_length = 20
replay_buffer = ExpBuffer(replay_buffer_size, sample_length)
batch_size = 64
eps_start = 0.9
eps = eps_start
eps_end = 0.05
eps_decay = 10
gamma = 0.999
learning_rate = 0.01
blind_prob = 0
EXPLORE = 300

adrqn = ADRQN(n_actions, state_size, embedding_size).cuda()
adrqn_target = ADRQN(n_actions, state_size, embedding_size).cuda()
adrqn_target.load_state_dict(adrqn.state_dict())

optimizer = torch.optim.Adam(adrqn.parameters(), lr=learning_rate)

for i_episode in range(M_episodes):
    now = time.time()
    done = False
    hidden = None
    last_action = 0
    current_return = 0
    last_observation = env.reset()
    for t in count():

        action, hidden = adrqn.act(torch.tensor(last_observation).float().view(1, 1, -1).cuda(), F.one_hot(
            torch.tensor(last_action), n_actions).view(1, 1, -1).float().cuda(), hidden=hidden, epsilon=eps)

        observation, reward, done, info = env.step(action)
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
        if i_episode > EXPLORE:
            eps = eps_end + (eps_start - eps_end) * \
                math.exp((-1*(i_episode-EXPLORE))/eps_decay)

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
            break

    print([[current_return], [eps]])
    adrqn_target.load_state_dict(adrqn.state_dict())

env.close()
