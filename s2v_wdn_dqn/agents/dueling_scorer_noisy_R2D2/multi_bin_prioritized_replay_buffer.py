import math
import numpy as np
import random
import torch
import itertools
from collections import namedtuple, deque

from s2v_wdn_dqn.agents.dueling_scorer_noisy_R2D2.sum_tree import SumTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EARLY_MEMORIES_SIZE = 8000
EXCEPTIONAL_MEMORIES_SIZE = 1000

Experience = namedtuple(
    "Experience",
    field_names=[
        "state", "edge_feature", "edge_status", "global_feats", "action", 
        "reward", "next_state", "next_edge_feature", "next_edge_status", "next_global_feats", "done",
        "episode", "step"]
)

class MultiBinPrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, store_sequence=False):
        """Initialize a PrioritizedReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.quarter_buffer_size = math.ceil(buffer_size // 4)
        self.quarter_batch_size = math.ceil(batch_size // 4)

        self.memory_good     = SumTree(self.quarter_buffer_size, self.quarter_batch_size)
        self.memory_half_good= SumTree(self.quarter_buffer_size, self.quarter_batch_size)
        self.memory_half_bad = SumTree(self.quarter_buffer_size, self.quarter_batch_size)
        self.memory_bad      = SumTree(self.quarter_buffer_size, self.quarter_batch_size)

        self.early_memories =       deque(maxlen=EARLY_MEMORIES_SIZE)      
        self.exceptional_memories = deque(maxlen=EXCEPTIONAL_MEMORIES_SIZE)

        self.alpha = 0.3  # prioritization exponent
        self.beta_start = 0.4   # importance-sampling exponent
        self.beta_end = 1.0
        self.beta_decay = buffer_size  # number of steps over which beta will be annealed
        self.beta = self.beta_start
        self.frame = 1  # to track the number of frames for beta annealing
        self.ready = False

        self.store_sequence = store_sequence
        if store_sequence:
            self.episodes = {}

    def is_ready(self):
        if not self.ready:
            self.ready = len(self.memory_good) >= self.quarter_batch_size and \
               len(self.memory_half_good) >= self.quarter_batch_size and \
               len(self.memory_half_bad) >= self.quarter_batch_size and \
               len(self.memory_bad) >= self.quarter_batch_size
            if self.ready:
                print("Multi-bin prioritized replay buffer is ready for sampling.")
        return self.ready

    def clear_buffer(self):
        self.memory_good     = SumTree(self.quarter_buffer_size, self.quarter_batch_size)
        self.memory_half_good= SumTree(self.quarter_buffer_size, self.quarter_batch_size)
        self.memory_half_bad = SumTree(self.quarter_buffer_size, self.quarter_batch_size)
        self.memory_bad      = SumTree(self.quarter_buffer_size, self.quarter_batch_size)
        
        self.early_memories.clear()
        self.exceptional_memories.clear()

    def add_new(self, state, edge_feature, edge_status, global_feats, action, reward, next_state, next_edge_feature, next_edge_status, next_global_feats, done, episode, step):
        assert type(action) == int or isinstance(action, np.integer), f"Action must be an integer, got {type(action)}"
        e = Experience(state, edge_feature, edge_status, global_feats, action, reward, next_state, next_edge_feature, next_edge_status, next_global_feats, done, episode, step)
        
        if (len(self) + 1) % 15000 == 0:
            n = len(self.early_memories)
            keep_fraction = 0.5
            keep = max(1, int(n * keep_fraction))
            start = n - keep
            self.early_memories = deque(itertools.islice(self.early_memories, start, None), maxlen=EARLY_MEMORIES_SIZE)
            print("Pruned early memories to size {}".format(len(self.early_memories)))

        if len(self.early_memories) < EARLY_MEMORIES_SIZE:
            self.early_memories.append(e)

        if reward >= 0.5:
            self.memory_good.add(1.0, e)
        elif 0.0 <= reward < 0.5:
            self.memory_half_good.add(1.0, e)
        elif -0.5 < reward < 0.0:
            self.memory_half_bad.add(1.0, e)
        else:  # reward <= -0.5
            self.memory_bad.add(1.0, e)

        if reward >= 1.0:
            self.exceptional_memories.append(e)

        if self.store_sequence:
            if episode not in self.episodes:
                self.episodes[episode] = {
                    "steps": [],
                    "final_reward": 0
                }
            self.episodes[episode]["steps"].append(e)
            if done:
                self.episodes[episode]["final_reward"] = sum(step.reward for step in self.episodes[episode]["steps"])

        if len(self) % 200 == 0:
            print(f"  Buffer sizes: good={self.memory_good.n_entries}, half_good={self.memory_half_good.n_entries}, half_bad={self.memory_half_bad.n_entries}, bad={self.memory_bad.n_entries}, early={len(self.early_memories)}, exceptional={len(self.exceptional_memories)}")

        #print("Added new experience with reward {:.4f} to buffer.".format(reward))

        #if self.memory_good.n_entries + self.memory_half_good.n_entries + self.memory_half_bad.n_entries + self.memory_bad.n_entries > 5000:
        #    #make histogram of rewards truncated to 2 decimal places
        #    all_rewards = []
        #    all_rewards += [e.reward for e in self.memory_good.data if e is not None]
        #    all_rewards += [e.reward for e in self.memory_half_good.data if e is not None]
        #    all_rewards += [e.reward for e in self.memory_half_bad.data if e is not None]
        #    all_rewards += [e.reward for e in self.memory_bad.data if e is not None]
        #    hist, bin_edges = np.histogram(all_rewards, bins=20)
        #    print("  Buffer size exceeded 5000, current reward histogram:")
        #    for count, edge in zip(hist, bin_edges):
        #        print(f"    {edge:.2f}: {count}")

        self.frame += 1
        self.beta = min(self.beta_end, self.beta_start + (self.frame / self.beta_decay) * (self.beta_end - self.beta_start))

    def update_experience(self, raw_exps, priorities):
        """Re-add an experience with a given priority to the appropriate buffer."""

        for (tag, exp, idx), prio in zip(raw_exps, priorities):
            new_p = float(np.clip(prio ** self.alpha, 1e-6, prio ** self.alpha))
            if tag == "good":        self.memory_good.update(idx, new_p)
            elif tag == "half_good": self.memory_half_good.update(idx, new_p)
            elif tag == "half_bad":  self.memory_half_bad.update(idx, new_p)
            elif tag == "bad":       self.memory_bad.update(idx, new_p)
            #else: raise ValueError(f"Unknown tag {tag}")

    #def update_priorities(self, indices, priorities):
    #    """Update the priorities of sampled experiences."""
    #    for idx, priority in zip(indices, priorities):
    #        new_p = np.clip(priority ** self.alpha, 1e-6, 1)
    #        self.memory_tree.update(idx, new_p)

    def sample(self, priority=True):
        """Sample a mixed batch from early, exceptional, and 4 PER bins with correct IS weights."""

        experiences, raw_exps, q_list = [], [], []

        #print("Sampling from buffer:")
        #print(f"  Early memories: {len(self.early_memories)}")
        #print(f"  Exceptional memories: {len(self.exceptional_memories)}")
        #print(f"  Good memories: {self.memory_good.n_entries}")
        #print(f"  Half-good memories: {self.memory_half_good.n_entries}")
        #print(f"  Half-bad memories: {self.memory_half_bad.n_entries}")
        #print(f"  Bad memories: {self.memory_bad.n_entries}")

        #prob_early = 0.10
        #prob_except = 0.10
        #bin_prob = 0.2

        perc_early = 10
        perc_extra = 10
        
        num_early = min(len(self.early_memories), self.batch_size // perc_early)
        if num_early > 0:
            indexes = random.sample(range(len(self.early_memories)), num_early)
            samp = [self.early_memories[i] for i in indexes]
            experiences += samp
            raw_exps += [("early", e, i) for e, i in zip(samp, indexes)]
            prob_early  = len(samp) / self.batch_size
            q_list += [prob_early / max(1, len(self.early_memories))] * len(samp)

        num_except = min(len(self.exceptional_memories), self.batch_size // perc_extra)
        if num_except > 0:
            indexes = random.sample(range(len(self.exceptional_memories)), num_except)
            samp = [self.exceptional_memories[i] for i in indexes]
            experiences += samp
            raw_exps += [("exceptional", e, i) for e, i in zip(samp, indexes)]
            prob_except = len(samp) / self.batch_size
            q_list += [prob_except / max(1, len(self.exceptional_memories))] * len(samp)

        n_remaining = self.batch_size - len(experiences)
        bins = [
            ("good",        self.memory_good),
            ("half_good",   self.memory_half_good),
            ("half_bad",    self.memory_half_bad),
            ("bad",         self.memory_bad),
        ]

        per_bin = {
            "good": int(0.4 * n_remaining),
            "half_good": int(0.2 * n_remaining),
            "half_bad": int(0.2 * n_remaining),
            "bad": int(0.2 * n_remaining),
        }

        for tag, tree in bins:
            if per_bin[tag] <= 0 or tree.n_entries == 0:
                continue
            exps, idxs, p_vals, probs, total_p, _ = tree.sample(per_bin[tag], self.beta) if priority else tree.uniform_sample(per_bin[tag]) #, self.beta)
            experiences += exps
            bin_prob = len(exps) / self.batch_size
            raw_exps += [(tag, e, idx) for e, idx in zip(exps, idxs)]
            for p in p_vals:
                q_i = bin_prob * (p / (total_p + 1e-12))
                q_list.append(q_i)

        n_remaining = self.batch_size - len(experiences)
        for i in range(n_remaining):
            tag, tree = bins[i % 4]
            if tree.n_entries == 0:
                continue
            exps, idxs, p_vals, probs, total_p, _ = tree.sample(1, self.beta) if priority else tree.uniform_sample(1) #, self.beta)
            experiences += exps
            bin_prob = 1.0 / self.batch_size
            raw_exps += [(tag, exps[0], idxs[0])]
            q_i = bin_prob * (p_vals[0] / (total_p + 1e-12))
            q_list.append(q_i)

        N = (
            len(self.early_memories)
            + len(self.exceptional_memories)
            + self.memory_good.n_entries
            + self.memory_half_good.n_entries
            + self.memory_half_bad.n_entries
            + self.memory_bad.n_entries
        )
        q_arr = np.asarray(q_list, dtype=np.float32)
        weights = ((1.0 / max(1, N)) / np.maximum(q_arr, 1e-12)) ** self.beta
        weights /= weights.max() + 1e-12
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

        # ---------- Convert experiences to tensors ----------
        states = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(device)
        edge_features = torch.from_numpy(np.stack([e.edge_feature for e in experiences])).float().to(device)
        edge_status = torch.from_numpy(np.stack([e.edge_status for e in experiences])).float().to(device)
        global_feats = torch.from_numpy(np.stack([e.global_feats for e in experiences])).float().to(device)

        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device).view(-1)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(device)
        next_edge_features = torch.from_numpy(np.stack([e.next_edge_feature for e in experiences])).float().to(device)
        next_edge_status = torch.from_numpy(np.stack([e.next_edge_status for e in experiences])).float().to(device)
        next_global_feats = torch.from_numpy(np.stack([e.next_global_feats for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return (
            states, edge_features, edge_status, global_feats,
            actions, rewards,
            next_states, next_edge_features, next_edge_status, next_global_feats,
            dones, weights, raw_exps
        )


    def sample_sequence(self, seq_length, priority=True):
        '''
        For LSTM training: sample sequences of experiences of given length.
        If not enough sequences can be formed, fall back to regular sampling.
        '''

        if not self.store_sequence:
            raise ValueError("Cannot sample sequences when store_sequence is False")

        (_, _, _, _, _, _, _, _, _, _, _, weights, raw_exps) = self.sample(priority=priority)

        seqs, w_list, handles = [], [], []
        for (tag, e_anchor, idx), w_anchor in zip(raw_exps, weights):
            episode = self.episodes.get(e_anchor.episode, [])
            ep_mem = episode["steps"]
            final_reward = episode["final_reward"]

            if not ep_mem: continue
            end = e_anchor.step + 1
            start = max(0, end - seq_length)
            seq = ep_mem[start:end]
            if len(seq) < seq_length: continue
            seqs.append(seq)
            w_list.append(w_anchor.item())
            handles.append((tag, seq[-1], idx))

        if not seqs:
            return None


        # Convert sequences to tensors of shape (B, T, …)
        def stack(attr):
            arr = np.stack([[getattr(e, attr) for e in seq] for seq in seqs]).astype(np.float32)
            return torch.tensor(arr, device=device)
        
        states = stack("state")
        edge_features = stack("edge_feature")
        edge_status = stack("edge_status")
        global_feats = stack("global_feats")
        actions = torch.tensor([[e.action for e in seq] for seq in seqs], dtype=torch.long, device=device)
        rewards = torch.tensor([[e.reward for e in seq] for seq in seqs], dtype=torch.float32, device=device)
        next_states = stack("next_state")
        next_edge_features = stack("next_edge_feature")
        next_edge_status = stack("next_edge_status")
        next_global_feats = stack("next_global_feats")
        dones = torch.tensor([[e.done for e in seq] for seq in seqs], dtype=torch.float32, device=device)

        w = torch.tensor(w_list, dtype=torch.float32, device=device).unsqueeze(1)        
        return (
            states, edge_features, edge_status, global_feats,
            actions, rewards,
            next_states, next_edge_features, next_edge_status, next_global_feats,
            dones, w, handles
    )


    def __len__(self):
        return (self.memory_good.n_entries + self.memory_half_good.n_entries +
            self.memory_half_bad.n_entries + self.memory_bad.n_entries)
            #len(self.early_memories) + len(self.exceptional_memories))
