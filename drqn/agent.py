import torch
import random
import numpy as np


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                # exploit
                return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(self.device)

    def act(self, observation, last_action, epsilon, policy_net, hidden=None):
        q_values, hidden_out = policy_net(
            observation, last_action, hidden)
        if np.random.uniform() > epsilon:
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.num_actions)
        return action, hidden_out
