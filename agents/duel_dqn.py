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
class DuelingDQN(nn.Module):
    """Light 32‑unit version with dropout (matches original snippet)."""

    def __init__(self, state_dim: int, action_dim: int, dim: int = 32, dropout: float = 0.35) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.v = nn.Linear(dim, 1)
        self.a = nn.Linear(dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(F.relu(self.fc2(x)))
        v = self.v(x)
        a = self.a(x)
        return v + a - a.mean(dim=1, keepdim=True)


# Dueling DQN Agent simply inherits DQNAgent
# class DuelingDQNAgent(DQNAgent):
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

class DDQNAgent:
    def __init__(self, state_dim, action_dim,tol, gamma=0.95, lr=1e-3, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.policy.state_dict())
        self.q_value_diffs = []  
        self.q_stable = False
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  
        self.train_steps = 0
        self.greedy_bias = 0.8
        self.tol = tol
        self.q_value_diffs_all=[]
        self.success_history = deque(maxlen=tol)  
        self.early_stop = False


    def update_success(self, done):
        self.success_history.append(done)
        success_rate = sum(self.success_history) / len(self.success_history)
        
        if len(self.success_history) <= self.tol/2:
            return
  
        if success_rate > 0.99:
                self.early_stop = True
        
    def take_action(self, state, deterministic=False):
        eps_threshold = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
                        np.exp(-1.0 * self.train_steps / self.epsilon_decay)
        self.train_steps += 1

        if not deterministic and random.random() < eps_threshold:
            if random.random() < self.greedy_bias:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_vals = self.policy(state_tensor)
                return q_vals.argmax().item()
            else:
                return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_vals = self.policy(state_tensor)
            return q_vals.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def soft_update(self, tau=0.005):
        for target_param, param in zip(self.target_model.parameters(), self.policy.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

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
            prev_q_values = self.policy(states).clone()

        with torch.no_grad():
            next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, next_actions).squeeze()
        q_target = rewards + self.gamma * (1 - dones) * next_q

        q_values = self.policy(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = nn.MSELoss()(q_pred, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            new_q_values = self.policy(states)
            diff = torch.mean((prev_q_values - new_q_values) ** 2).item()
            self.q_value_diffs.append(diff)
            self.q_value_diffs_all.append(diff)

            if len(self.q_value_diffs) > 10:
                self.q_value_diffs.pop(0)
                avg_q_change = sum(self.q_value_diffs) / len(self.q_value_diffs)
                self.q_stable = avg_q_change < 1e-4

        self.train_steps += 1
        self.epsilon *= self.epsilon_decay
        if self.epsilon <= self.epsilon_min + 1e-4:
            self.epsilon = self.epsilon_min

        self.soft_update()
        
