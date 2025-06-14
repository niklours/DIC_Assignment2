import numpy as np
from agents.ppo_agent import PPOAgent
from new_environment import ContinuousSpace
import matplotlib.pyplot as plt
import argparse  

def train_ppo(episodes=500, max_steps=100):
    env = ContinuousSpace(width=10.0, height=10.0)
    agent = PPOAgent(state_dim=10, action_dim=8)

    episode_rewards = []

    for ep in range(episodes):
        env = ContinuousSpace(width=10.0, height=10.0)
        env.add_object(8.0, 8.0, 1.0, "target")
        env.place_agent(1.0, 1.0, size=1.0)

        total_reward = 0
        for step in range(max_steps):
            state = env.get_state_vector()
            action = agent.take_action(state)
            reward = env.step_with_reward(action)

            next_state = env.get_state_vector()
            agent.update(next_state, reward, action)

            total_reward += reward
            if env.is_task_complete():
                break

        agent.finish_episode()
        episode_rewards.append(total_reward)

        if ep % 10 == 0:
            print(f"Episode {ep}: Total Reward = {total_reward:.2f}")

    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Reward Curve")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train PPO Agent")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    args = parser.parse_args()

    train_ppo(episodes=args.episodes, max_steps=args.steps)
