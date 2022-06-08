import gym
import torch


class EnvManager():
    def __init__(self, device, env_name):
        self.device = device
        self.env = gym.make(env_name).unwrapped
        self.env.reset()
        self.done = False
        self.current_state = None

    def reset(self):
        self.current_state = self.env.reset()

    def take_action(self, action):
        return self.env.step(action)

    def num_state_features(self):
        return self.env.observation_space.shape[0]

    def get_state(self):
        return self.current_state

    def close(self):
        self.env.close()

    def num_actions_available(self):
        return self.env.action_space.n
