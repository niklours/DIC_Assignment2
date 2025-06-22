import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):  # 增加神经元数
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        shared = self.shared(state)
        return self.policy_head(shared), self.value_head(shared)

class PPOAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, tol=300, gamma=0.9, lr=1e-4, clip_eps=0.4,
                 entropy_coef=0.8, value_coef=0.6, lam=0.95, batch_size=64, ppo_epochs=4):
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.tol = tol
        self.lam = lam

        self.reset_buffers()

    def reset_buffers(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def take_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.actor_critic(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.probs.argmax().item() if deterministic else dist.sample().item()

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(torch.tensor(action).to(self.device)))
        self.values.append(value.squeeze().item())
        return action

    def store_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_and_advantages(self, next_value=0):
        values = self.values + [next_value]
        returns, advantages = [], []
        gae = 0
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        return returns, advantages

    def update(self):
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).detach()
        returns, advantages = self.compute_returns_and_advantages()
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            probs, values = self.actor_critic(states)
            dist = torch.distributions.Categorical(probs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (returns - values.squeeze()).pow(2).mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.reset_buffers()
