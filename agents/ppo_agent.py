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

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.95, eps_clip=0.2, K_epochs=8):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, _ = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def evaluate(self, states, actions):
        logits, state_values = self.policy(states)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy
    
    # Storing previous experience
    def store(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def update(self, memory):
        states = torch.FloatTensor(memory['states']).to(self.device)
        actions = torch.LongTensor(memory['actions']).to(self.device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(self.device)
        returns = torch.FloatTensor(memory['returns']).to(self.device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(states, actions)

            advantages = returns - state_values.detach()
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - state_values).pow(2).mean() - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.__init__()

    def compute_returns(self, gamma):
        returns = []
        R = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + gamma * R
            returns.insert(0, R)
        return returns

