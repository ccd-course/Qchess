import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class ADRQN(nn.Module):
    def __init__(self, n_actions, state_size, embedding_size):
        super(ADRQN, self).__init__()
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.embedder = nn.Linear(n_actions, embedding_size)
        self.obs_layer = nn.Linear(state_size, 16)
        self.obs_layer2 = nn.Linear(16, 32)
        self.lstm = nn.LSTM(input_size=32 + embedding_size,
                            hidden_size=128, batch_first=True)
        self.out_layer = nn.Linear(128, n_actions)

    def forward(self, observation, action, hidden=None):
        # Takes observations with shape (batch_size, seq_len, state_size)
        # Takes one_hot actions with shape (batch_size, seq_len, n_actions)
        action_embedded = self.embedder(action)
        observation = F.relu(self.obs_layer(observation))
        observation = F.relu(self.obs_layer2(observation))
        lstm_input = torch.cat([observation, action_embedded], dim=-1)
        if hidden is not None:
            lstm_out, hidden_out = self.lstm(lstm_input, hidden)
        else:
            lstm_out, hidden_out = self.lstm(lstm_input)

        q_values = self.out_layer(lstm_out)
        return q_values, hidden_out
