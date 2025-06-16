import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        # shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        # value stream V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # advantage stream A(s,a)
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        x = self.shared(x)
        v = self.value_head(x)
        a = self.advantage_head(x)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

class DuelingDQNAgent:
    def __init__(self,
                 state_dim, action_dim,
                 lr=1e-4, gamma=0.99,
                 buffer_size=100_000, batch_size=64,
                 target_update=50,
                 eps_start=1.0, eps_end=0.2, eps_decay=100_000,
                 device='cpu'):
        
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # networks
        self.q_net = DuelingQNetwork(state_dim, action_dim).to(device)
        self.tgt_net = DuelingQNetwork(state_dim, action_dim).to(device)
        self.tgt_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr) 

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=100,
                                                   gamma=0.5)

        self.replay = ReplayBuffer(buffer_size)

        # epsilon for exploration
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.steps_done = 0

    def select_action(self, state):
        # compute current epsilon
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # epsilon-greedy: random vs. best
        if np.random.rand() < eps:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state_v = torch.tensor(state, dtype=torch.float32, device=self.device)
            q_vals = self.q_net(state_v.unsqueeze(0))
            return int(q_vals.argmax().item())

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = \
            self.replay.sample(self.batch_size)

        #convert to tensors
        states_v = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_v = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_v = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # calc current Q(s,a)
        q_vals = self.q_net(states_v).gather(1, actions_v).squeeze(1)

        # computing target q
        with torch.no_grad():
            next_q = self.tgt_net(next_states_v).max(1)[0]
            target_q = rewards_v + self.gamma * next_q * (1 - dones_v)

        loss = torch.nn.functional.mse_loss(q_vals, target_q)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.tgt_net.load_state_dict(self.q_net.state_dict())
