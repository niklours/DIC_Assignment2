import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.policy_head(x), self.value_head(x)


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        k_epochs=4,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.model(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def compute_gae(self, rewards, values, dones, next_value):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_update(self, memory, batch_size=64):
        states = torch.FloatTensor(np.array(memory["states"])).to(self.device)
        actions = torch.LongTensor(memory["actions"]).to(self.device)
        old_logprobs = torch.FloatTensor(memory["logprobs"]).to(self.device)
        returns = torch.FloatTensor(memory["returns"]).to(self.device)
        advantages = torch.FloatTensor(memory["advantages"]).to(self.device)

        for _ in range(self.k_epochs):
            idx = np.arange(states.size(0))
            np.random.shuffle(idx)
            for start in range(0, states.size(0), batch_size):
                end = start + batch_size
                batch_idx = idx[start:end]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                probs, values = self.model(batch_states)
                dist = Categorical(probs)
                entropy = dist.entropy().mean()
                new_logprobs = dist.log_prob(batch_actions)

                ratio = (new_logprobs - batch_old_logprobs).exp()
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.view(-1), batch_returns)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
