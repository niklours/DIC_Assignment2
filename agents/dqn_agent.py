import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.8
        self.train_steps = 0
        self.update_target_every = 100

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.model(state)
        return q_vals.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        q_target = rewards + self.gamma * (1 - dones) * next_q_values.max(dim=1)[0]
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = nn.MSELoss()(q_pred, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        
        self.train_steps += 1
        if self.train_steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())