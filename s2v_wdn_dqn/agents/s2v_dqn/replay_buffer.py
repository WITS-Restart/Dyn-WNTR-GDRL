import numpy as np
import random
import torch
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Experience = namedtuple(
    "Experience",
    field_names=["state", "edge_feature", "action", "reward", "next_state", "next_edge_feature", "done"]
)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def clear_buffer(self):
        self.memory = deque(maxlen=self.buffer_size)

    def add(self, state, edge_feature, action, reward, next_state, next_edge_feature, done):
        """Add a new experience to memory."""
        e = Experience(state, edge_feature, action, reward, next_state, next_edge_feature, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        sample_size = min(self.batch_size, len(self.memory))
        experiences = random.sample(self.memory, k=sample_size)

        # vstack for single-valued (0-dimensional) elements and stack for n-dimensional elements
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        edge_features = torch.from_numpy(np.stack([e.edge_feature for e in experiences if e is not None])).float().to(device)
        # states = [torch.from_numpy(e.state).float().to(device) for e in experiences if e is not None]
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_edge_features = torch.from_numpy(np.stack([e.next_edge_feature for e in experiences if e is not None])).float().to(device)
        # next_states = [torch.from_numpy(e.next_state).float().to(device) for e in experiences if e is not None]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, edge_features, actions, rewards, next_states, next_edge_features, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
