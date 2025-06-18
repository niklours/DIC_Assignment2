import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
from agents.dqn_agent import DQNAgent

# Dueling Network Architecture
# class DuelingDQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.feature = nn.Sequential(
#             nn.Linear(input_dim, 256), nn.ReLU(),
#             nn.Linear(256, 128), nn.ReLU(),
#             nn.Linear(128, 64), nn.ReLU()
#         )
#         self.value_stream = nn.Sequential(
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Linear(32, 1)
#         )
#         self.advantage_stream = nn.Sequential(
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Linear(32, output_dim)
#         )

#     def forward(self, x):
#         x = self.feature(x)
#         value = self.value_stream(x)
#         advantage = self.advantage_stream(x)
#         q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
#         return q_vals

# Duelling Deep Q Network that estimates Value function
class DuellingDQN(nn.Module):
    def __init__(self, state_size, action_size, dim=32, dropout_rate=0.35):
        super(DuellingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.A = nn.Linear(dim, action_size)
        #self.dropout = nn.Dropout(p=dropout_rate)  # Initializing dropout layer
        # estimate value function for duelling DQN
        self.V = nn.Linear(dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # get the estimated q function 
        A = self.A(x)
        V = self.V(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

# Dueling DQN Agent simply inherits DQNAgent
class DuelingDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, gamma=0.95, lr=1e-3, batch_size=32):
        super().__init__(state_dim, action_dim, gamma, lr, batch_size)
        
