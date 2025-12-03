import numpy as np
import random
import torch
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


Experience = namedtuple(
    "Experience",
    field_names=["state", "edge_feature", "edge_status", "global_feats", "action", "reward", "next_state", "next_edge_feature", "next_edge_status", "next_global_feats", "done"]
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

    def add(self, state, edge_feature, edge_status, global_feats, action, reward, next_state, next_edge_feature, next_edge_status, next_global_feats, done):
        """Add a new experience to memory."""
        #action = np.array(action, dtype=np.int16)
        #assert action.shape == (2,), f"Action must be [i,j], got {action.shape}"
        assert type(action) == int or isinstance(action, np.integer), f"Action must be an integer, got {type(action)}"
        e = Experience(state, edge_feature, edge_status, global_feats, action, reward, next_state, next_edge_feature, next_edge_status, next_global_feats, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        sample_size = min(self.batch_size, len(self.memory))
        experiences = random.sample(self.memory, k=sample_size)

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

        return states, edge_features, edge_status, global_feats, actions, rewards, next_states, next_edge_features, next_edge_status, next_global_feats, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
