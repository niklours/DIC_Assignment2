import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
from torch.distributions import Categorical
from new_environment import ContinuousSpace 



class BackboneNetwork(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dimensions),
            nn.Tanh(),
            nn.LayerNorm(hidden_dimensions),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.Tanh(),
            nn.LayerNorm(hidden_dimensions),
            nn.Linear(hidden_dimensions, out_features),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.net(x)
        return self.dropout(x)
    
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64, latent_dim=64, dropout=0.1):
        super(ActorCritic, self).__init__()
        self.backbone = BackboneNetwork(input_dim, hidden_dim, latent_dim, dropout)
        self.actor_head = nn.Linear(latent_dim, action_dim)
        self.critic_head = nn.Linear(latent_dim, 1)

    def forward(self, state):
        latent = self.backbone(state)
        policy_logits = self.actor_head(latent)
        value = self.critic_head(latent)
        return policy_logits, value.squeeze(-1)

def forward(self, state):
        x = self.shared(state)
        logits = self.policy_layer(x)
        value = self.value_layer(x)
        return logits, value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = Memory(capacity=10000)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, _ = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob.item()

    def store(self, state, action, logprob, reward, done):
        self.memory.store(state, action, logprob, reward, done)

    def update(self, epochs=4, batch_size=64):
        states, actions, logprobs, rewards, dones = self.memory.get_all()
        returns = self.memory.compute_returns(self.gamma)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_logprobs = torch.FloatTensor(logprobs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        for _ in range(epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_returns = returns[batch_idx]

                logits, values = self.policy(batch_states)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                advantages = batch_returns - values.squeeze()

                ratios = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory.clear()


from collections import deque

class Memory:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, logprob, reward, done):
        self.buffer.append((state, action, logprob, reward, done))

    def clear(self):
        self.buffer.clear()

    def compute_returns(self, gamma):
        returns = []
        R = 0
        rewards = [t[3] for t in self.buffer]
        dones = [t[4] for t in self.buffer]

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + gamma * R
            returns.insert(0, R)

        return returns

    def get_all(self):
        states = [t[0] for t in self.buffer]
        actions = [t[1] for t in self.buffer]
        logprobs = [t[2] for t in self.buffer]
        rewards = [t[3] for t in self.buffer]
        dones = [t[4] for t in self.buffer]
        return states, actions, logprobs, rewards, dones