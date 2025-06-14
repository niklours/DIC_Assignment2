import argparse
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from agents.ppo_agent import PPOAgent
from new_environment import ContinuousSpace
from img_gen import get_grid_image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_env():
    env = ContinuousSpace(width=10.0, height=10.0)
    env.add_object(8.0, 8.0, 1.0, "target")
    env.place_agent(1.0, 1.0, size=1.0)
    return env

def train_ppo(episodes, max_steps, gamma, clip_eps, lr, entropy_coef, seed):
    set_seed(seed)
    agent = PPOAgent(
        state_dim=10,
        action_dim=8,
        gamma=gamma,
        clip_eps=clip_eps,
        lr=lr,
        entropy_coef=entropy_coef,
    )

    rewards_per_episode = []
    best_reward = float("-inf")
    best_env = None

    for ep in range(episodes):
        env = setup_env()
        total_reward = 0

        for step in range(max_steps):
            state = env.get_state_vector()
            action = agent.take_action(state)
            reward = env.step_with_reward(action)
            agent.update(state, reward, action)
            total_reward += reward
            if env.is_task_complete():
                break

        agent.finish_episode()
        rewards_per_episode.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_env = env

        if ep % 10 == 0:
            print(f"Episode {ep}: Total Reward = {total_reward:.2f}")

    # Plot reward curve
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Reward Curve")
    plt.grid(True)
    plt.show()

    # Show final path
    if best_env:
        plt.imshow(get_grid_image(best_env))
        plt.title("Final Path After PPO Training")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_ppo(
        episodes=args.episodes,
        max_steps=args.steps,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
    )
