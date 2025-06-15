import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
from agents.dqn_agent import DQNAgent

# Dueling Network Architecture
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

# Dueling DQN Agent simply inherits DQNAgent
class DuelingDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, gamma=0.95, lr=1e-3, batch_size=32):
        super().__init__(state_dim, action_dim, gamma, lr, batch_size)
