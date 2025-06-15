import argparse
import matplotlib.pyplot as plt
from img_gen import get_grid_image
from new_environment import ContinuousSpace  
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent, Memory
import sys
import os
import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn as nn
directions = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']

def setup_env():
    world = ContinuousSpace(width=11.0, height=11.0, wall_size=1.0)
    world.add_object(8.0, 6.0, 1.0, "target")
    #world.add_object(4.0, 6.0, 1.0, "target")
    obstacle_coords = [
        (3.0, 3.0, 3.2, 3.2),
        (3.0, 2.3, 3.2, 2.4),
        (7.0, 7.0, 8.0, 8.0),     
        (2.0, 6.0, 3.5, 7.0),     
    ]
    for x1, y1, x2, y2 in obstacle_coords:
        world.add_rectangle_object(x1, y1, x2, y2, size=1.0, obj_type="obstacle")

    world.place_agent(2.0, 2.0, 0.6)
    return world


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent in ContinuousSpace environment.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode.")
    args = parser.parse_args()

    state_dim = 7
    action_dim = len(directions)
    agent = PPOAgent(state_dim, action_dim)
    memory = Memory()
    completed_flags = []
    rewards_list = []

    update_interval = 5  # PPO update every 5 episodes

    for episode in range(args.episodes):
        env = setup_env()
        state = env.get_state_vector()
        total_reward = 0

        for step in range(args.steps):
            action_idx, logprob = agent.select_action(state)
            reward = env.step_with_reward(action_idx, step_size=0.2, sub_step=0.05)
            next_state = env.get_state_vector()
            done = env.is_task_complete()

            memory.states.append(state)
            memory.actions.append(action_idx)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.dones.append(done)
            #agent.store(state, action_idx, reward, next_state, done)

            state = next_state
            total_reward += reward
            if done:
                break

        completed_flags.append(done)
        rewards_list.append(total_reward)

        if (episode + 1) % update_interval == 0:
            returns = memory.compute_returns(agent.gamma)
            memory_dict = {
                'states': memory.states,
                'actions': memory.actions,
                'logprobs': memory.logprobs,
                'returns': returns
            }
            agent.update(memory_dict)
            memory.clear()

        print(f"[Episode {episode+1}] Total reward: {total_reward:.2f}")

    max_rew = -(np.inf)
    for _ in range(1):
        env = setup_env()
        state = env.get_state_vector()
        total_reward = 0
        steps = 0
        while not env.is_task_complete() and steps < args.steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            logits, _ = agent.policy(state_tensor)
            action_idx = torch.argmax(logits, dim=-1).item()

            reward = env.step_with_reward(action_idx, step_size=0.2, sub_step=0.05)
            state = env.get_state_vector()
            total_reward += reward
            steps += 1

        print(f"[Evaluation] Reward: {total_reward:.2f} | Steps: {steps}")
        if total_reward > max_rew:
            max_rew = total_reward
            best_env = env

    plt.imshow(get_grid_image(best_env))
    plt.title("Final Path After Training")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.violinplot(y=rewards_list)
    plt.title("Training Reward Distribution (Violin Plot)")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

