import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from s2v_wdn_dqn.agents.s2v_dqn.model import QNetwork
from s2v_wdn_dqn.agents.s2v_dqn.replay_buffer import ReplayBuffer

BUFFER_SIZE = 5000        # replay buffer size
BATCH_SIZE = 64           # minibatch size
GAMMA = 1.00              # discount factor
TAU = 5e-3                # for soft update of target parameters
LR = 1e-4                 # learning rate
CLIP_GRAD_NORM_VALUE = 5  # value of gradient to clip while training
UPDATE_TARGET_EACH = 500  # number of steps to wait until updating target network
UPDATE_PARAMS_EACH = 4    # number of steps to wait until sampling experience tuples and updating model params
WARMUP_STEPS = 1000       # number of steps to wait before start learning

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent():
    """Interacts with and learns from the environment."""

    _SUPPORTED_PROBLEMS = {'mvc', 'tsp'}

    def __init__(
        self,
        problem,
        n_node_features,
        n_edge_features,
        nstep=1,
        embedding_dim=64,
        embedding_layers=4,
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
        double_dqn=False,
    ):
        """Initialize an Agent object"""
        super().__init__(problem)

        self.nstep = nstep
        self.use_nstep = nstep > 1
        self.double_dqn = double_dqn

        self.gamma = gamma
        self.clip_grad_norm_value = clip_grad_norm_value
        self.update_target_each = update_target_each
        self.update_params_each = update_params_each
        self.warmup_steps = warmup_steps
        self.tau = tau
        self.target_update = target_update
        assert target_update in ("soft", "hard"), 'target_update must be one of {"soft", "hard"}'

        # Q-Network
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.qnetwork_local = QNetwork(embed_dim=embedding_dim, embedding_layers=embedding_layers, n_node_features=n_node_features,
                                       n_edge_features=n_edge_features, normalize=normalize).to(device, dtype=torch.float32)
        self.qnetwork_target = QNetwork(embed_dim=embedding_dim, embedding_layers=embedding_layers, n_node_features=n_node_features,
                                        n_edge_features=n_edge_features, normalize=normalize).to(device, dtype=torch.float32)

        self.hard_update(self.qnetwork_local, self.qnetwork_target)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.global_t_step = 0
        self.update_t_step = 0

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # To be used in n-step learning
        self.gamma_n_minus_1 = self.gamma ** (self.nstep - 1)
        self.gamma_n = self.gamma ** self.nstep

        # Internal values accross steps
        self.episode_losses = []
        self.losses = []
        self.q_expecteds = []
        self.q_targets = []
        self.params = []
        # self.grads = []
        self.theta1s = []
        self.theta2s = []
        self.theta3s = []
        self.theta4s = []
        self.theta5s = []
        self.theta6s = []
        self.theta7s = []

        # torch.nn.init.xavier_normal_(self.qnetwork_local.embedding_layer.theta1.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.embedding_layer.theta2.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.embedding_layer.theta3.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.embedding_layer.theta4.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.q_layer.theta5.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.q_layer.theta6.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.q_layer.theta7.weight)

        # Initial episode config
        self.reset_episode()

    def reset_episode(self):
        self.espisode_t_step = 0
        self.states = deque(maxlen=self.nstep)
        self.edge_features = deque(maxlen=self.nstep)
        self.actions = deque(maxlen=self.nstep)
        self.rewards = deque(maxlen=self.nstep)
        self.sum_rewards = 0

        if self.qnetwork_local.training:
            self.losses.append(sum(self.episode_losses) / len(self.episode_losses) if len(self.episode_losses) > 0 else 0)
            self.episode_losses = []

    @torch.no_grad()
    def act(self, state, edge_feature, *args, **kwargs):
        """Returns actions for given state as per current policy.

        Params
        ======
            obs        : current observation
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).to(device, dtype=torch.float32)
        edge_feature = torch.from_numpy(edge_feature).to(device, dtype=torch.float32)

        # Valid actions are nodes that aren´t already in the partial solution
        xv = state[:, 0]
        valid_actions = (xv == 0).nonzero()

        # Epsilon-greedy: greedy action selection
        eps = kwargs.get("eps", 0.0)
        if random.random() < eps:
            action_idx = np.random.randint(len(valid_actions))
            action = valid_actions[action_idx].item()
            return action

        action_values = self.qnetwork_local(state, edge_feature).squeeze(0)  # squeeze to remove NN batching
        # print(f"{action_values=}")
        # print(f"{action_values.shape=}")
        # print(f"{valid_actions=}")
        # print(f"{xv=}")
        valid_actions_idx = action_values[valid_actions].argmax().item()
        action = valid_actions[valid_actions_idx].item()
        return action

    def step(self, state, edge_feature, action, reward, next_state, next_edge_feature, done):
        self.espisode_t_step += 1
        self.global_t_step += 1
        if not self.use_nstep:
            # Save experience in replay memory
            self.memory.add(state, edge_feature, action, reward, next_state, next_edge_feature, done)
            if len(self.memory) >= BATCH_SIZE \
                    and self.global_t_step % self.update_params_each == 0 \
                    and self.global_t_step >= self.warmup_steps:
                experiences = self.memory.sample()
                self.learn(experiences)
        else:
            reward_to_subtract = self.rewards[0] if self.espisode_t_step > self.nstep else 0  # r1

            self.states.append(state)
            self.edge_features.append(edge_feature)
            self.actions.append(action)
            self.rewards.append(reward)

            if len(self.rewards) <= self.nstep:
                self.sum_rewards += reward * (self.gamma ** (len(self.rewards)-1))
            else:
                # reward_to_subtract = self.rewards[0]
                self.sum_rewards = (self.sum_rewards - reward_to_subtract) / self.gamma
                self.sum_rewards += reward * self.gamma_n_minus_1

            # Get xv from info
            if self.espisode_t_step >= self.nstep:
                # Get oldest state and action (S_{t-n}, a_{t-n}) to add to replay memory buffer
                oldest_state = self.states[0]
                oldest_edge_feature = self.edge_features[0]
                oldest_action = self.actions[0]

                # Save experience in replay memory
                self.memory.add(
                    oldest_state,
                    oldest_edge_feature,
                    oldest_action,
                    self.sum_rewards,
                    next_state,
                    next_edge_feature,
                    done
                )

            # Different from the paper, as it should be called inside (if self.t_step >= self.nstep)
            if len(self.memory) >= BATCH_SIZE \
                    and self.global_t_step % self.update_params_each == 0 \
                    and self.global_t_step >= self.warmup_steps:
                experiences = self.memory.sample()
                self.learn(experiences)

    # def get_valid_actions(self, states):
    #     actions = [torch.nonzero(state == 0).view(-1) for state in states]
    #     return [x if x.shape[0] > 0 else torch.tensor([0]) for x in actions]

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        """
        states, edge_features, actions, rewards, next_states, next_edge_features, dones = experiences

        xv = next_states[:, :, 0]
        invalid_actions_mask = (xv == 1)

        # For Double DQN, first get actions that maximize Q_next using local network,
        # then use those actions to get Q values using target network
        if self.double_dqn:
            with torch.no_grad():
                local_preds = self.qnetwork_local(next_states, next_edge_features).detach()
                next_actions = local_preds.masked_fill(invalid_actions_mask, -1e18).max(1, True)[1]

                target_preds = self.qnetwork_target(next_states, next_edge_features).detach()
                q_targets_next = target_preds.gather(1, next_actions)
        else:
            target_preds = self.qnetwork_target(next_states, next_edge_features).detach()

            # Calculate q_targets_next for valid actions
            # target_preds.shape = (batch_size, n_vertices)
            with torch.no_grad():
                q_targets_next = target_preds.masked_fill(invalid_actions_mask, -1e18).max(1, True)[0]

        # Calc q_targets based on Q_targets_next
        q_targets = rewards + self.gamma_n * q_targets_next * (1 - dones)

        # Calculate Q value
        q_expected = self.qnetwork_local(states, edge_features).gather(1, actions)

        # Calc loss
        # loss = F.mse_loss(q_expected, q_targets)
        loss = F.huber_loss(q_expected, q_targets)

        self._log_params(actions, dones, invalid_actions_mask, loss, next_states, q_expected, q_targets, states, target_preds)

        # Run optimizer step
        self.optimizer.zero_grad()
        # self.losses.append(loss.item())
        self.episode_losses.append(loss.item())
        loss.backward()
        # self.grads.append(next(self.qnetwork_local.parameters())[0,0].grad)
        # print("theta1 grad:", self.qnetwork_local.embedding_layer.theta1.weight.grad)
        # Gradient clipping to avoid exploding gradient
        if self.clip_grad_norm_value is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        elif self.target_update == "hard":
            self.update_t_step = (self.update_t_step + 1) % self.update_target_each
            if self.update_t_step == 0:
                # print(f"{self.global_t_step=}")
                self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def _log_params(self, actions, dones, invalid_actions_mask, loss, next_states, q_expected, q_targets, states, target_preds):
        if loss.item() > 5e150:
            print(f'actions: {list(actions.cpu().detach().numpy().flatten())}')
            print(f'q_expected: {list(q_expected.cpu().detach().numpy().flatten())}')
            print(f'q_targets: {list(q_targets.cpu().detach().numpy().flatten())}')
            print(f'{loss=}')
            print(f'{target_preds=}')
            print(f'{invalid_actions_mask=}')
            print(f'{dones=}')
            print(f'{1 - dones=}')
            print(f'{states=}')
            print(f'{next_states=}')
            print(f'{self.espisode_t_step=}')
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
        checkpoint = torch.load(checkpoint_path)
        self.global_t_step = checkpoint['global_t_step']
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_model(self, checkpoint_path):
        """Save model's checkpoint"""
        checkpoint = {
            'global_t_step': self.global_t_step,
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
