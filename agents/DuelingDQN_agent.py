
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np


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
