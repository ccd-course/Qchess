import torch
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
from . import qchess_env
from epsilon_greedy_strategy import EpsilonGreedyStrategy
from agent import Agent
from experience_replay import ReplayMemory, Experience, extract_tensors
from network import Network, QValues
from plot import plot, get_moving_average


# Hyperparameters
batch_size = 256
gamma = 0.999  # Discount factor
eps_start = 1  # Exploration rate: start value.
eps_end = 0.01  # Exploration rate: end value
eps_decay = 0.001  # Exploration rate: decay value
target_update = 10  # Interval for updating the target net
memory_size = 100000  # Capacity of the replay memory
lr = 0.001  # Learning rate
num_episodes = 1000  # Num of episode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = qchess_env.env()
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, env.action_space_size, device)
memory = ReplayMemory(memory_size)

policy_net = Network(env.state_space_size,
                     env.action_space_size).to(device)
target_net = Network(env.state_space_size,
                     env.action_space_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)


episode_durations = []

for episode in range(num_episodes):
    env.reset()
    state = env.observe()
    for timestep in count():
        action = agent.select_action(state, policy_net)
        # TODO: change to receive returns of step (not just reward)
        reward = env.step(action)
        next_state = env.observe()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(
                experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # TODO: change to something like env.done (idk if we have that)
        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if get_moving_average(100, episode_durations)[-1] >= 195:
        break
env.close()
