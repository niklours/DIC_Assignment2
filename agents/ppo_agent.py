import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, action_dim), nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        shared = self.shared(state)
        return self.policy_head(shared), self.value_head(shared)


class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        tol=300,
        gamma=0.99,
        lr=3e-4,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        batch_size=256,
        ppo_epochs=10,
        gae_lambda=0.95,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.tol = tol
        self.gae_lambda = gae_lambda
        self.entropy_coef_init = entropy_coef  # Initial entropy coef for annealing
        self.entropy_coef_final = 0.001  # Minimum entropy coef after annealing
        self.entropy_anneal_episodes = 500  # Number of episodes over which to anneal
        self.episode_count = 0  # Track episodes for annealing

        self.reset_buffers()

    def reset_buffers(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def policy(self, state_tensor):
        """Return only action probabilities for compatibility with helper.py"""
        probs, _ = self.actor_critic(state_tensor)
        return probs

    def take_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.actor_critic(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.probs.argmax().item() if deterministic else dist.sample().item()

        if not deterministic:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(dist.log_prob(torch.tensor(action).to(self.device)))
            self.values.append(value.flatten().item())
        return action

    def store_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_and_advantages(self, next_value=0):
        values = self.values + [next_value]
        returns, advantages = [], []
        gae = 0
        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + self.gamma * values[step + 1] * (1 - self.dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        return returns, advantages

    def anneal_entropy_coef(self):
        # Linearly anneal entropy coefficient from initial to final value
        frac = min(self.episode_count / self.entropy_anneal_episodes, 1.0)
        self.entropy_coef = (
            self.entropy_coef_init * (1 - frac) + self.entropy_coef_final * frac
        )

    def update(self):
        if not self.states:
            return
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).detach()
        last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = (
                self.actor_critic(last_state)[1].item() if not self.dones[-1] else 0.0
            )
        returns, advantages = self.compute_returns_and_advantages(next_value)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        batch_size = self.batch_size if self.batch_size > 0 else dataset_size

        self.anneal_entropy_coef()

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                probs, values = self.actor_critic(mb_states)
                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(mb_actions)

                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * mb_advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_returns - values.flatten()).pow(2).mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), max_norm=0.5
                )
                self.optimizer.step()

        self.reset_buffers()
        self.episode_count += 1  # Increment episode count for entropy annealing
