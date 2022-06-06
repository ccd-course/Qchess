import numpy as np
import torch


class ExpBuffer():
    # Alternative Experience Buffer that stores sequences of fixed length
    def __init__(self, max_seqs, seq_len):
        self.max_seqs = max_seqs
        self.counter = 0
        self.seq_len = seq_len
        self.storage = [[] for i in range(max_seqs)]

    def write_tuple(self, aoaro):
        if len(self.storage[self.counter]) >= self.seq_len:
            self.counter += 1
        self.storage[self.counter].append(aoaro)

    def sample(self, batch_size):
        # Sample batches of (action, observation, action, reward, observation, done) tuples
        # With dimensions (batch_size, seq_len) for rewards/actions/done and (batch_size, seq_len, obs_dim) for observations
        last_actions = []
        last_observations = []
        actions = []
        rewards = []
        observations = []
        dones = []

        for i in range(batch_size):
            seq_idx = np.random.randint(self.counter)
            last_act, last_obs, act, rew, obs, done = zip(
                *self.storage[seq_idx])
            last_actions.append(list(last_act))
            last_observations.append(last_obs)
            actions.append(list(act))
            rewards.append(list(rew))
            observations.append(list(obs))
            dones.append(list(done))

        return torch.tensor(last_actions).cuda(), torch.tensor(np.array(last_observations), dtype=torch.float32).cuda(), torch.tensor(actions).cuda(), torch.tensor(rewards).float().cuda(), torch.tensor(np.array(observations), dtype=torch.float32).cuda(), torch.tensor(dones).cuda()
