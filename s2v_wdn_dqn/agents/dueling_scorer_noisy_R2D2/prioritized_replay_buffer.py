import math
import numpy as np
import random
import torch
from collections import namedtuple, deque

from s2v_wdn_dqn.agents.dueling_scorer_noisy_R2D2.sum_tree import SumTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


Experience = namedtuple(
    "Experience",
    field_names=["state", "edge_feature", "edge_status", "global_feats", "action", "reward", "next_state", "next_edge_feature", "next_edge_status", "next_global_feats", "done"]
)

PriorityzedExperience = namedtuple(
    "PriorityzedExperience",
    field_names=["priority", "experience"]
)

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a PrioritizedReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory_tree = SumTree(buffer_size, batch_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.alpha = 0.3  # prioritization exponent
        self.beta_start = 0.4   # importance-sampling exponent
        self.beta_end = 1.0
        self.beta_decay = buffer_size  # number of steps over which beta will be annealed
        self.beta = self.beta_start
        self.frame = 1  # to track the number of frames for beta annealing

    def clear_buffer(self):
        self.memory_tree = SumTree(self.buffer_size, self.batch_size)

    def add(self, state, edge_feature, edge_status, global_feats, action, reward, next_state, next_edge_feature, next_edge_status, next_global_feats, done):
        """Add a new experience to memory."""
        #action = np.array(action, dtype=np.int16)
        #assert action.shape == (2,), f"Action must be [i,j], got {action.shape}"
        assert type(action) == int or isinstance(action, np.integer), f"Action must be an integer, got {type(action)}"
        priority = 1.0 # New experience gets max priority
        e = Experience(state, edge_feature, edge_status, global_feats, action, reward, next_state, next_edge_feature, next_edge_status, next_global_feats, done)
        self.memory_tree.add(priority, e)

        self.frame += 1
        self.beta = min(self.beta_end, self.beta_start + (self.frame / self.beta_decay) * (self.beta_end - self.beta_start))

    def update_priorities(self, indices, priorities):
        """Update the priorities of sampled experiences."""
        for idx, priority in zip(indices, priorities):
            new_p = np.clip(priority ** self.alpha, 1e-6, 1)
            self.memory_tree.update(idx, new_p)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if random.random() < 2:
            indexes, experiences, weights = self.memory_tree.uniform_sample()
        else:
            indexes, experiences, weights = self.memory_tree.sample(self.beta)
        
        # vstack for single-valued (0-dimensional) elements and stack for n-dimensional elements
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        edge_features = torch.from_numpy(np.stack([e.edge_feature for e in experiences if e is not None])).float().to(device)
        edge_status   = torch.from_numpy(np.stack([e.edge_status for e in experiences])).float().to(device)        # (B,E)
        global_feats = torch.from_numpy(np.stack([e.global_feats for e in experiences if e is not None])).float().to(device)

        # states = [torch.from_numpy(e.state).float().to(device) for e in experiences if e is not None]
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device).view(-1)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_edge_features = torch.from_numpy(np.stack([e.next_edge_feature for e in experiences if e is not None])).float().to(device)
        next_edge_status   = torch.from_numpy(np.stack([e.next_edge_status for e in experiences])).float().to(device)
        next_global_feats = torch.from_numpy(np.stack([e.next_global_feats for e in experiences if e is not None])).float().to(device)
        # next_states = [torch.from_numpy(e.next_state).float().to(device) for e in experiences if e is not None]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, edge_features, edge_status, global_feats, actions, rewards, next_states, next_edge_features, next_edge_status, next_global_feats, dones, indexes, weights

    def __len__(self):
        """Return the current size of internal memory."""
        return self.memory_tree.n_entries
