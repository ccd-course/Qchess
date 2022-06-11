import math
import torch
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qchess_env

class EnvManager():
    def __init__(self, device):
        self.device = device
        self.env = qchess_env.env()
        self.env.reset()
        self.done = False
        self.current_state = None
        self.current_mask = None

    def format_state(self, input):
        flattened = input.flatten()
        return flattened

    def reset(self):
        self.env.reset()
        cs, _, _, _ = self.env.last()
        self.current_state = self.format_state(cs["observation"])
        self.current_mask = self.format_state(cs["action_mask"])

    def take_action(self, action):
        self.env.step(action.item())
        cs, reward, _, _ = self.env.last()
        self.current_state = self.format_state(cs["observation"])
        self.current_mask = self.format_state(cs["action_mask"])
        return torch.tensor([reward], device=self.device)

    def num_state_features(self):
        state_size = math.prod(self.env.observation_spaces["player_0"].spaces["observation"].shape)
        return state_size

    def get_state(self):
        if self.done:
            return np.zeros_like(
                self.current_state
            )
        else:
            return np.array(self.current_state)

    def get_mask(self):
        if self.done:
            return np.zeros_like(
               self.current_mask
            )
        else:
            return np.array(self.current_mask)


    def num_actions_available(self):
        return self.env.action_spaces["player_0"].n
