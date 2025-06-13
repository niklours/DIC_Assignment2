
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals


# Inherit DQNAgent and just replace model init
from agents.dqn_agent import DQNAgent

class DuelingDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        # Replace the networks with DuelingDQN
        self.model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Update optimizer to optimize the new model parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs.get('lr', 1e-3))
