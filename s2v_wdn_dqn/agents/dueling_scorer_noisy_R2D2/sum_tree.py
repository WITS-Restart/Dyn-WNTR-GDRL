import random
import numpy as np


class SumTree:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.empty(capacity, dtype=object)
        self.n_entries = 0
        self.batch_size = batch_size

    def add(self, p, data):
        p = float(np.nan_to_num(p, nan=1e-6, posinf=1e6, neginf=1e-6))
        p = np.clip(p, 1e-6, 1)
        if self.n_entries < self.capacity:
            data_idx = self.n_entries
            tree_idx = self.capacity - 1 + data_idx
        else:
            leaf_start = self.capacity - 1
            min_leaf_offset = np.argmin(self.tree[leaf_start:])
            tree_idx = leaf_start + min_leaf_offset
            data_idx = min_leaf_offset

        self.data[data_idx] = data
        self.update(tree_idx, p)
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        p = float(np.nan_to_num(p, nan=1e-6, posinf=1e6, neginf=1e-6))
        p = max(p, 1e-6)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
        
    def remove(self, idx):
        """Zero out leaf idx (absolute index in tree array)."""
        # idx is the absolute index in self.tree
        if idx < self.capacity - 1 or idx >= len(self.tree):
            raise IndexError("Not a leaf index")
        change = -self.tree[idx]
        self.tree[idx] = 0.0
        self._propagate(idx, change)


    def sample(self, n, beta):
        batch, idxs, priorities = [], [], []
        segment = self.total / n
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            idx = self._retrieve(0, s)
            data_idx = idx - (self.capacity - 1)
            if self.data[data_idx] is None:
                continue 
            batch.append(self.data[data_idx])
            idxs.append(idx)
            priorities.append(self.tree[idx])

        total_p = self.tree[0] + 1e-12
        probs = [p / total_p for p in priorities]

        # IS weights (local, within tree)
        sampling_prob = np.array(probs)
        weights = (self.n_entries * sampling_prob) ** (-beta)
        weights /= weights.max() + 1e-12

        return batch, idxs, priorities, probs, total_p, weights

    def uniform_sample(self, n):
        """Uniformly sample n items. Match sample() return signature."""
        batch, idxs, priorities = [], [], []

        n_avail = self.n_entries
        if n_avail == 0:
            return batch, idxs, priorities, [], 1e-12, np.array([], dtype=np.float32)

        replace = n > n_avail
        data_indices = np.random.choice(n_avail, n, replace=replace)

        for data_idx in data_indices:
            leaf_idx = (self.capacity - 1) + int(data_idx)
            if self.data[data_idx] is None:
                continue
            batch.append(self.data[data_idx])
            idxs.append(leaf_idx)
            priorities.append(1.0)

        k = max(1, n_avail)
        probs = [1.0 / k] * len(batch)

        weights = np.ones(len(batch), dtype=np.float32)
        total_p = float(k)
        return batch, idxs, priorities, probs, total_p, weights



    @property
    def total(self):
        return self.tree[0]
    
    def __len__(self):
        return self.n_entries
