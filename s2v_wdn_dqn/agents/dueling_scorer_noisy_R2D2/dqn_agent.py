import csv
import math
import random
import sys
from collections import deque
from wsgiref import handlers

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from global_state import State
from logger import DEFAULT_LOGGER_KEY, LogType, Logger
from s2v_wdn_dqn.agents.dueling_scorer_noisy_R2D2.multi_bin_prioritized_replay_buffer import MultiBinPrioritizedReplayBuffer
from .model import QNetwork, NoisyLinear


BUFFER_SIZE = 200000            # replay buffer size
BATCH_SIZE = 256                # minibatch size
GAMMA = 0.99 #0.995 #1.00       # discount factor
TAU = 5e-3                      # for soft update of target parameters
LR =  5e-5                      # learning rate
CLIP_GRAD_NORM_VALUE = 3        #4  # value of gradient to clip while training
UPDATE_TARGET_EACH = 10000       # number of steps to wait until updating target network
UPDATE_PARAMS_EACH = 1          # number of steps to wait until sampling experience tuples and updating model params
UPDATE_SCHEDULER_EACH = 10000    # number of steps to wait until updating learning rate scheduler
WARMUP_STEPS = 0             # number of steps to wait before start learning
GLOBAL_DIM = 5                  # number of global features: (angry clients, frac closed, wrong iso, sources stranded, sensors percentage)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        n_node_features,
        n_edge_features,
        embedding_dim,
        embedding_layers,
        nstep=1,
        normalize=True,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        lr=LR,
        clip_grad_norm_value=CLIP_GRAD_NORM_VALUE,
        update_target_each=UPDATE_TARGET_EACH,
        target_update="soft",
        update_params_each=UPDATE_PARAMS_EACH,
        warmup_steps=WARMUP_STEPS,
        double_dqn=True,
        partial_observability=False,
    ):

        self.nstep = nstep
        self.use_nstep = nstep > 1
        self.double_dqn = double_dqn

        self.gamma = gamma
        self.clip_grad_norm_value = clip_grad_norm_value
        self.update_params_each = update_params_each
        self.update_target_each = update_target_each
        self.warmup_steps = warmup_steps
        self.tau = tau

        #self.target_update = target_update
        #assert target_update in ("soft", "hard"), 'target_update must be one of {"soft", "hard"}'

        # Q-Network
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.qnetwork_local = QNetwork(embed_dim=embedding_dim, global_dim=GLOBAL_DIM, embedding_layers=embedding_layers, n_node_features=n_node_features,
                                       n_edge_features=n_edge_features, normalize=normalize).to(device, dtype=torch.float32)
        self.qnetwork_target = QNetwork(embed_dim=embedding_dim, global_dim=GLOBAL_DIM, embedding_layers=embedding_layers, n_node_features=n_node_features,
                                        n_edge_features=n_edge_features, normalize=normalize).to(device, dtype=torch.float32)

        self.hard_update(self.qnetwork_local, self.qnetwork_target)
        #self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)

        self.global_t_step = 0
        self.update_t_step = 0

        # Replay memory
        #self.memory = ReplayBuffer(buffer_size, batch_size)
        #self.memory = PrioritizedReplayBuffer(buffer_size, batch_size)
        self.memory = MultiBinPrioritizedReplayBuffer(buffer_size, batch_size)

        # To be used in n-step learning
        self.gamma_n_minus_1 = self.gamma ** (self.nstep - 1)
        self.gamma_n = self.gamma ** self.nstep

        # Internal values accross steps
        self.q_expecteds = []
        self.q_targets = []
        self.params = []

        self.episode_losses = []
        self.losses = []
        self.grads = []

        # Initial episode config
        self.reset_episode()

    def reset_episode(self):
        self.episode_t_step = 0
        self.states = deque(maxlen=self.nstep)
        self.edge_features = deque(maxlen=self.nstep)
        self.edge_status = deque(maxlen=self.nstep)
        self.global_features = deque(maxlen=self.nstep)
        self.actions = deque(maxlen=self.nstep)
        self.rewards = deque(maxlen=self.nstep)
        self.steps = deque(maxlen=self.nstep)
        self.sum_rewards = 0

        if self.qnetwork_local.training:
            self.losses.append(sum(self.episode_losses) / len(self.episode_losses) if len(self.episode_losses) > 0 else 0)
            self.episode_losses = []

    def set_valid_edges(self, valid_edges):
        """
        valid_edges: np.ndarray or list of shape (E, 2) with node indices for each controllable edge
        """
        self.valid_edges = np.asarray(valid_edges, dtype=np.int64)
        self.E = self.valid_edges.shape[0]
        self.no_op_index = self.E
        # Cache edges_ij on the right device
        self.edges_ij = torch.as_tensor(self.valid_edges, dtype=torch.long, device=device)

    @torch.no_grad()
    def act(self, state, edge_feature, edge_status, global_feats, training: bool):
        """
        Returns an int in [0..E], where E is the no-op.
        """

        state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
        ef_t    = torch.from_numpy(edge_feature).unsqueeze(0).float().to(device)
        es_t = torch.from_numpy(edge_status).unsqueeze(0).float().to(device)
        gl_t = torch.from_numpy(global_feats).unsqueeze(0).float().to(device)

        flow_idx = 0  # index of flow feature in edge features
        flows_t = ef_t[0, :, flow_idx]          

        if not training: 
            self.eval()
            for net in [self.qnetwork_local, self.qnetwork_target]:
                for m in net.modules():
                    if isinstance(m, NoisyLinear):
                        m.remove_noise()
        else:   
            self.train()
            for m in self.qnetwork_local.modules():
                if isinstance(m, NoisyLinear):
                    m.sample_noise()
            
        q = self.qnetwork_local(state_t, ef_t, self.edges_ij, edge_status=es_t, global_feats=gl_t)
        q = q.squeeze(0)

        if training:
            abs_flows = torch.abs(flows_t)
            min_flow = abs_flows.min()
            max_flow = abs_flows.max()
            ranges = max_flow - min_flow + 1e-6

            b1 = min_flow + 0.25 * ranges
            b2 = min_flow + 0.50 * ranges
            b3 = min_flow + 0.75 * ranges

            bins = [
                abs_flows[abs_flows <= b1],
                abs_flows[(abs_flows > b1) & (abs_flows <= b2)],
                abs_flows[(abs_flows > b2) & (abs_flows <= b3)],
                abs_flows[abs_flows > b3],
            ]

            if bins[1].numel() > 0:
                threshold = bins[1].min()
            elif bins[2].numel() > 0:
                threshold = bins[2].min()
            elif bins[0].numel() > 0:
                threshold = bins[0].max()
            else:
                threshold = bins[3].min()
            
            if self.global_t_step < self.warmup_steps:
                min_flow_for_action = threshold
            elif self.global_t_step < self.warmup_steps + 10000:
                min_flow_for_action = threshold * 0.1
            else:
                min_flow_for_action = 0.0
                
            closed_t = (es_t[0] < 0.5)  # (E,) 1=open, 0=closed
            suggested_t = (torch.abs(flows_t) >= min_flow_for_action) | closed_t 
            if suggested_t.any():
                q_masked = q.masked_fill(~suggested_t, -1e6)
            else:
                q_masked = q

            if self.global_t_step < self.warmup_steps:
                suggested_idx = torch.nonzero(suggested_t, as_tuple=False).view(-1)
                if suggested_idx.numel() > 0:
                    return int(suggested_idx[torch.randint(len(suggested_idx), (1,))].item())
                else:
                    return int(torch.randint(self.E, (1,)).item())
        else:
            q_masked = q         

        return int(torch.argmax(q_masked).item())

    def step(self, state, edge_feature, edge_status, global_feats, action, reward, next_state, next_edge_feature, next_edge_status, next_global_feats, done, episode, step):
        self.episode_t_step += 1
        self.global_t_step += 1
        if not self.use_nstep:
            self.memory.add_new(state, edge_feature, edge_status, global_feats, action, reward, next_state, next_edge_feature, next_edge_status, next_global_feats, done, episode, step)
            
            #print("len memory:", len(self.memory), self.memory.buffer_size, self.memory.batch_size, self.global_t_step, self.update_params_each, self.warmup_steps)
            if len(self.memory) >= self.memory.batch_size \
                    and self.global_t_step % self.update_params_each == 0 \
                    and self.global_t_step >= self.warmup_steps:
                self.learn(priority=True)
        else:

            self.states.append(state)
            self.edge_features.append(edge_feature)
            self.actions.append(action)
            self.rewards.append(reward)
            self.edge_status.append(edge_status)
            self.global_features.append(global_feats)
            self.steps.append(step)

            #reward_to_subtract = self.rewards[0] if self.episode_t_step > self.nstep else 0  # r1
            #if len(self.rewards) <= self.nstep:
            #    self.sum_rewards += reward * (self.gamma ** (len(self.rewards)-1))
            #else:
            #    self.sum_rewards = self.gamma * (self.sum_rewards - reward_to_subtract) + reward * (self.gamma ** (self.nstep-1))


            window = list(self.rewards)
            self.sum_rewards = sum(r * (self.gamma ** i) for i, r in enumerate(window))
            after_n_steps = self.episode_t_step >= self.nstep

            # Get xv from info
            if after_n_steps:
                # Get oldest state and action (S_{t-n}, a_{t-n}) to add to replay memory buffer
                oldest_state = self.states[0]
                oldest_edge_feature = self.edge_features[0]
                oldest_action = self.actions[0]
                oldest_edge_status = self.edge_status[0]
                oldest_global_feats = self.global_features[0]
                oldest_step = self.steps[0]

                #print("Adding n-step experience at global step", self.global_t_step, " Sum rewards:", self.sum_rewards)

                self.memory.add_new(
                    oldest_state,
                    oldest_edge_feature,
                    oldest_edge_status,
                    oldest_global_feats,
                    oldest_action,
                    self.sum_rewards,
                    next_state,
                    next_edge_feature,
                    next_edge_status,
                    next_global_feats,
                    done,
                    episode,
                    oldest_step
                )
            
            if done:
                start_from = 0 if not after_n_steps else 1 # already pushed oldest step or not

                rewards = list(self.rewards)  # deque -> list for stable slicing
                for index in range(start_from, len(rewards)):
                    remaining_reward = 0.0
                    for i, r in enumerate(rewards[index:]):
                        remaining_reward += (self.gamma ** i) * r

                    #print("Adding final n-step experience at global step", self.global_t_step, " Sum rewards:", remaining_reward)

                    self.memory.add_new(
                        self.states[index],
                        self.edge_features[index],
                        self.edge_status[index],
                        self.global_features[index],
                        self.actions[index],
                        remaining_reward,
                        next_state,
                        next_edge_feature,
                        next_edge_status,
                        next_global_feats,
                        True, #False if index + 1 < len(rewards) else True,
                        episode,
                        self.steps[index]
                    )

            # Different from the paper, as it should be called inside (if self.t_step >= self.nstep)
            if len(self.memory) >= self.memory.batch_size \
                    and self.global_t_step % self.update_params_each == 0 \
                    and self.global_t_step >= self.warmup_steps:
                #print("Learning step at global step", self.global_t_step)
                #print("len memory:", len(self.memory), self.memory.buffer_size, self.memory.batch_size, self.global_t_step, self.update_params_each, self.warmup_steps)
                self.learn(priority=True)

    def learn(self, priority=True):
        (states, edge_features, edge_status, global_features,
        actions, rewards, next_states, next_edge_features, next_edge_status, 
        next_global_features, dones, weights, raw_exps) = self.memory.sample(priority=priority)

        actions   = actions.view(-1)                           # (B,)
        B         = actions.size(0)
        batch_idx = torch.arange(B, device=actions.device)


        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        for m in self.qnetwork_local.modules():
            if isinstance(m, NoisyLinear): m.sample_noise()
        for m in self.qnetwork_target.modules():
            if isinstance(m, NoisyLinear): m.remove_noise()

        q_pred= self.qnetwork_local(states, edge_features, self.edges_ij, edge_status, global_features)   # (B, E+1)
        q_expected = q_pred[batch_idx, actions].unsqueeze(1)                    # (B, 1)

        with torch.no_grad():
            target_q = self.qnetwork_target(next_states, next_edge_features, self.edges_ij, next_edge_status, next_global_features)  # (B, E+1)

            if self.double_dqn:
                local_q  = self.qnetwork_local(next_states, next_edge_features, self.edges_ij, next_edge_status, next_global_features)  # (B, E+1)
                a_star   = local_q.argmax(dim=1)                                                # (B,)
                q_t_next = target_q[batch_idx, a_star].unsqueeze(1)                             # (B,1)
            else:
                q_t_next = target_q.max(dim=1, keepdim=True)[0]                                 # (B,1)
        
        #q_targets = rewards + (self.gamma * q_t_next * (1 - dones))                           # (B,1)

        discount = self.gamma_n if self.use_nstep else self.gamma
        q_targets = rewards + (discount * q_t_next * (1 - dones))



        td_errors = q_expected - q_targets
        per_sample_loss = F.smooth_l1_loss(q_expected, q_targets, reduction="none").squeeze()  # (B,)
        weights_t = weights.to(per_sample_loss.device, dtype=torch.float32)
        loss = (weights_t * per_sample_loss).mean()
        
        #loss = F.huber_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        self.episode_losses.append(loss.item())
        loss.backward()

        with torch.no_grad():
            step = self.global_t_step
            State.get(DEFAULT_LOGGER_KEY).log(f"{step}, {loss.item()}", msg_type=LogType.LOSS)
            
            td_err_mean = td_errors.abs().mean().item()
            State.get(DEFAULT_LOGGER_KEY).log(f"{step}, {td_err_mean}", msg_type=LogType.TD_ERRORS)

            q_mean = q_pred.mean().item()
            q_std = q_pred.std().item()
            State.get(DEFAULT_LOGGER_KEY).log(f"{step}, {q_mean}, {q_std}", msg_type=LogType.Q_VALUES)

            total_grad_norm = 0.0
            for p in self.qnetwork_local.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            State.get(DEFAULT_LOGGER_KEY).log(f"{self.global_t_step}, {total_grad_norm}", msg_type=LogType.GRAD_NORM)

        if self.clip_grad_norm_value is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        
        self.optimizer.step()
        #self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau) #soft update every step
        self.update_t_step += 1                                                #hard update every update_target_each steps
        if self.update_t_step % self.update_target_each == 0:
            self.hard_update(self.qnetwork_local, self.qnetwork_target)
        if self.global_t_step % UPDATE_SCHEDULER_EACH == 0:
            self.scheduler.step()

        
        new_priorities = td_errors.detach().abs().cpu().numpy() + 1e-5
        self.memory.update_experience(raw_exps, new_priorities)


    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def hard_update(local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local

        Inputs:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        target_model.load_state_dict(local_model.state_dict())

    def train(self):
        """Configure PyTorch modules to be in train mode"""
        self.qnetwork_target.train()
        self.qnetwork_local.train()

    def eval(self):
        """Configure PyTorch modules to be in eval mode"""
        self.qnetwork_target.eval()
        self.qnetwork_local.eval()

    def load_model(self, checkpoint_path):
        """Load model's checkpoint"""
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.global_t_step = 0 #checkpoint['global_t_step']
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_model(self, checkpoint_path):
        """Save model's checkpoint"""
        print(f"Saving model to {checkpoint_path}")
        checkpoint = {
            'global_t_step': self.global_t_step,
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)





''' --- IGNORE ---

def _log_params(self, actions, dones, loss, next_states, q_expected, q_targets, states, target_preds):
        if loss.item() > 5e150:
            print(f'actions: {list(actions.cpu().detach().numpy().flatten())}')
            print(f'q_expected: {list(q_expected.cpu().detach().numpy().flatten())}')
            print(f'q_targets: {list(q_targets.cpu().detach().numpy().flatten())}')
            print(f'{loss=}')
            print(f'{target_preds=}')
            print(f'{dones=}')
            print(f'{1 - dones=}')
            print(f'{states=}')
            print(f'{next_states=}')
            print(f'{self.episode_t_step=}')
            sys.exit(0)
        self.q_targets.append(q_targets.min().item())
        self.q_expecteds.append(q_expected.min().item())
        # self.params.append(next(self.qnetwork_local.parameters())[0,0].item())
        self.theta1s.append(self.qnetwork_local.node_features_embedding_layer.theta1.weight[0, 0].item())
        self.theta2s.append(self.qnetwork_local.embedding_layer.theta2.weight[0, 0].item())
        self.theta3s.append(self.qnetwork_local.edge_features_embedding_layer.theta3.weight[0, 0].item())
        # self.theta4s.append(self.qnetwork_local.edge_features_embedding_layer.theta4.weight[0,0].item())
        self.theta5s.append(self.qnetwork_local.q_layer.theta5.weight[0, 0].item())
        self.theta6s.append(self.qnetwork_local.q_layer.theta6.weight[0, 0].item())
        self.theta7s.append(self.qnetwork_local.q_layer.theta7.weight[0, 0].item())



    @torch.no_grad()
    def old_act(self, state, edge_feature, *args, **kwargs):
        """Returns actions for given state as per current policy.

        Params
        ======
            obs        : current observation
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
        ef_t    = torch.from_numpy(edge_feature).unsqueeze(0).float().to(device)

        if USE_NEW_EDGE_Q_LAYER:
            qmat = self.qnetwork_local(state_t, ef_t, self.edges_ij).squeeze(0)  # (E+1,)
        else:
            qmat = self.qnetwork_local(state_t, ef_t).squeeze(0)  # torch.Tensor (N*N+1)
        
        M     = qmat.shape[0]  # M = N*N + 1
        E     = len(self.valid_edges)
        N = int(math.sqrt(M - 1))  # N*N+1 = M, so N = sqrt(M-1)
        no_op_idx = M - 1  # no-op action is the last one

        idx = None
        #print(f"M={M}, E={E}, qmat.shape={qmat.shape}")

        eps = kwargs.get("eps", 0.0)
        if random.random() < eps:
            p_noop = 0.3
            if random.random() < p_noop:
                print(f"Random No-op action selected - {E}")
                idx = E
            else: 
                idx = np.random.randint(E)
                print(f"Random action selected ={idx}")
        else:
            edge_qs = qmat[self.valid_edges_linear]  # shape [E]
            best_edge_pos = int(edge_qs.argmax())   # 0..E-1
            if qmat[ no_op_idx ] > edge_qs[best_edge_pos]:
                print(f"## Agent No-op action selected - {no_op_idx}")
                idx = E  #the no-op index
            else:
                idx = best_edge_pos  
                print(f"## Agent action selected - {idx} (edge {self.valid_edges[idx]})")

        return int(idx)

        
        #state = torch.from_numpy(state).to(device, dtype=torch.float32)
        #edge_feature = torch.from_numpy(edge_feature).to(device, dtype=torch.float32)
        ## adjacency lives in your `state` matrix after the first n_node_features columns
        #
        #valid_edges = list(self.edge_map.keys()) 
        #eps = kwargs.get("eps", 0.0)
        #if random.random() < eps:
        #    idx = np.random.randint(len(valid_edges))
        #    return valid_edges[idx].tolist()  # returns [i, j]
        #action_values = self.qnetwork_local(state, edge_feature).squeeze(0)  # (N, N)
        #scores = action_values[valid_edges[:, 0], valid_edges[:, 1]]
        #best_idx = scores.argmax()
        #best_action = valid_edges[best_idx].tolist()
        #return best_action

    @torch.no_grad()
    def old_act(self, state, edge_feature, *args, **kwargs):
        """Returns actions for given state as per current policy.

        Params
        ======
            obs        : current observation
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).to(device, dtype=torch.float32)
        edge_feature = torch.from_numpy(edge_feature).to(device, dtype=torch.float32)

        # Valid actions are nodes that aren´t already in the partial solution
        #xv = state[:, 0]
        #valid_actions = (xv == 0).nonzero()
        #action_values = self.qnetwork_local(state, edge_feature).squeeze(0)  # squeeze to remove NN batching
        #valid_actions_idx = action_values[valid_actions].argmax().item()
        #action = valid_actions[valid_actions_idx].item()
        #return action

        edge_state = edge_feature[0, :, :, 0]  # e.g., 1=open, 0=closed
        valid_actions = edge_state.nonzero(as_tuple=False)


        # Epsilon-greedy: greedy action selection
        eps = kwargs.get("eps", 0.0)
        if random.random() < eps:
            action_idx = np.random.randint(len(valid_actions))
            action = valid_actions[action_idx].item()
            return action

        action_values = self.qnetwork_local(state, edge_feature).squeeze(0)  # shape: (N, N)
        valid_scores = action_values[valid_actions[:, 0], valid_actions[:, 1]]
        best_idx = valid_scores.argmax()
        best_action = valid_actions[best_idx].tolist()  # returns [i, j]
        return best_action

    def old_old_learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) 
        """
        states, edge_features, actions, rewards, next_states, next_edge_features, dones = experiences
        batch_size = actions.size(0)

        # -------------------- Q(s,a) --------------------
        q_pred = self.qnetwork_local(states, edge_features)  # (B, N*N +1)
        #i, j = actions[:, 0], actions[:, 1]
        #batch_indices = torch.arange(batch_size, device=actions.device)
        batch_indices = torch.arange(batch_size, device=actions.device)
        q_expected = q_pred[batch_indices, actions.squeeze(-1)].unsqueeze(1)  # (B, 1)

        # -------------------- Q-target --------------------
        with torch.no_grad():
            if self.double_dqn:
                # a_max = argmax_a Q_local(s', a)
                local_q = self.qnetwork_local(next_states, next_edge_features)  # (B, N, N)
                a_max = local_q.view(batch_size, -1).argmax(dim=1)  # index in flattened N x N
                a_max_i = a_max // local_q.shape[2]
                a_max_j = a_max % local_q.shape[2]

                # Q_target(s', a_max)
                target_q = self.qnetwork_target(next_states, next_edge_features)  # (B, N, N)
                q_targets_next = target_q[batch_indices, a_max_i, a_max_j].unsqueeze(1)  # (B, 1)
            else:
                target_q = self.qnetwork_target(next_states, next_edge_features)  # (B, N, N)
                q_targets_next = target_q.view(batch_size, -1).max(dim=1, keepdim=True)[0]  # (B, 1)

        # TD target
        q_targets = rewards + self.gamma_n * q_targets_next * (1 - dones)

        # -------------------- Loss --------------------
        loss = F.huber_loss(q_expected, q_targets)

        self._log_params(actions, dones, loss, next_states, q_expected, q_targets, states, target_q)

        self.optimizer.zero_grad()
        self.episode_losses.append(loss.item())
        loss.backward()
        if self.clip_grad_norm_value is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        self.optimizer.step()

        # -------------------- Target network update --------------------
        if self.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        elif self.target_update == "hard":
            self.update_t_step = (self.update_t_step + 1) % self.update_target_each
            if self.update_t_step == 0:
                self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def old_learn(self, experiences):
        states, edge_features, actions, rewards, next_states, next_edge_features, dones = experiences
        actions = actions.view(-1)              # (B,)
        B = actions.size(0)
        batch_idx = torch.arange(B, device=actions.device)

        # Forward
        q_pred = self.qnetwork_local(states, edge_features)   # (B, M=N*N+1)
        M = q_pred.size(1)
        no_op_idx = M - 1

        # Map replay "edge-index or no-op" -> absolute column in q_pred
        # self.valid_edges_linear is set by env (shape [E]), keep it on the same device:
        edge_lin = self.valid_edges_linear.to(actions.device)  # (E,)
        E = edge_lin.size(0)

        is_noop = (actions == E)
        edge_pos = torch.clamp(actions, max=E-1)              # for non-noop rows
        a_abs = torch.where(is_noop, torch.full_like(actions, no_op_idx), edge_lin[edge_pos])

        q_expected = q_pred[batch_idx, a_abs].unsqueeze(1)    # (B,1)

        # ---- Targets: mask to valid edges + no-op ----
        with torch.no_grad():
            target_q_all = self.qnetwork_target(next_states, next_edge_features)  # (B,M)

            # Gather only valid columns
            target_edge = target_q_all[:, edge_lin]                                # (B,E)
            target_noop = target_q_all[:, no_op_idx].unsqueeze(1)                  # (B,1)
            target_masked = torch.cat([target_edge, target_noop], dim=1)           # (B,E+1)

            if self.double_dqn:
                local_q_all = self.qnetwork_local(next_states, next_edge_features) # (B,M)
                local_edge = local_q_all[:, edge_lin]                               # (B,E)
                local_noop = local_q_all[:, no_op_idx].unsqueeze(1)                 # (B,1)
                local_masked = torch.cat([local_edge, local_noop], dim=1)           # (B,E+1)

                a_rel = local_masked.argmax(dim=1)                                  # 0..E (E == no-op)
                a_abs_next = torch.where(a_rel == E, torch.full((B,), no_op_idx, device=actions.device), edge_lin[a_rel])
                q_targets_next = target_q_all[batch_idx, a_abs_next].unsqueeze(1)
            else:
                q_targets_next = target_masked.max(dim=1, keepdim=True)[0]

        q_targets = rewards + (self.gamma_n * q_targets_next * (1 - dones))

        loss = F.huber_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        self.episode_losses.append(loss.item())
        loss.backward()
        if self.clip_grad_norm_value is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        self.optimizer.step()

        # target update
        if self.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        else:
            self.update_t_step = (self.update_t_step + 1) % self.update_target_each
            if self.update_t_step == 0:
                self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def old_old_learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) 
        """
        states, edge_features, actions, rewards, next_states, next_edge_features, dones = experiences

        actions = actions.view(-1)  # shape: (batch_size,)
        batch_size = actions.size(0)
        batch_idx = torch.arange(batch_size, device=actions.device)

        # -------------------- Q(s,a) --------------------
        # Q-network outputs shape (batch, M) where M = N*N + 1
        q_pred = self.qnetwork_local(states, edge_features)  # (B, M)
        # Gather the Q-values corresponding to the taken actions
        q_expected = q_pred[batch_idx, actions].unsqueeze(1)  # (B, 1)

        # -------------------- Q-target --------------------
        with torch.no_grad():
            target_q = self.qnetwork_target(next_states, next_edge_features)  # (B, M)
            if self.double_dqn:
                # Double DQN: action selection by local network, evaluation by target network
                local_q_next = self.qnetwork_local(next_states, next_edge_features)  # (B, M)
                a_max = local_q_next.argmax(dim=1)  # (B,)
                q_targets_next = target_q[batch_idx, a_max].unsqueeze(1)  # (B, 1)
            else:
                # Standard DQN: max over next-state Q-values
                q_targets_next = target_q.max(dim=1, keepdim=True)[0]  # (B, 1)

        # Compute TD target
        q_targets = rewards + (self.gamma_n * q_targets_next * (1 - dones))  # (B, 1)

        # -------------------- Loss --------------------
        loss = F.huber_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        self.episode_losses.append(loss.item())
        loss.backward()

        # Gradient clipping
        if self.clip_grad_norm_value is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        self.optimizer.step()

        # -------------------- Target network update --------------------
        if self.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        else:
            self.update_t_step = (self.update_t_step + 1) % self.update_target_each
            if self.update_t_step == 0:
                self.hard_update(self.qnetwork_local, self.qnetwork_target)
'''
