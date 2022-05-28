import gym
import torch


class CartPoleEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.done = False
        self.current_state = None

    def reset(self):
        self.current_state = self.env.reset()

    def take_action(self, action):
        self.current_state, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def num_state_features(self):
        return self.env.observation_space.shape[0]

    def get_state(self):
        if self.done:
            return torch.zeros_like(
                torch.tensor(self.current_state), device=self.device
            ).float()
        else:
            return torch.tensor(self.current_state, device=self.device).float()

    def num_actions_available(self):
        return self.env.action_space.n
