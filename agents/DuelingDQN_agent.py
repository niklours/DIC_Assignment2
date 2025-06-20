import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from agents.dqn_agent import DQNAgent

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)

        self.model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.policy = self.model  # for compatibility with code expecting policy

        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs.get('lr', 1e-3))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        self.epsilon_start = kwargs.get('epsilon_start', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, next_actions).squeeze()

        q_target = rewards + self.gamma * (1 - dones) * next_q
        q_values = self.model(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = nn.MSELoss()(q_pred, q_target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.train_steps += 1
        self.soft_update()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)
