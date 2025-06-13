import pygame
import random
import argparse
import matplotlib.pyplot as plt
from img_gen import get_grid_image
from new_environment import ContinuousSpace
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
from agents.dqn_agent import DQNAgent

directions = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']

def setup_env():
    world = ContinuousSpace(width=11.0, height=11.0, wall_size=1.0)

    world.add_object(5.0, 5.0, 1.0, "target")
    world.add_rectangle_object(3.0, 3.0, 5.2, 5.4, size=2.0, obj_type="obstacle")
    world.add_rectangle_object(7.0, 7.0, 9.2, 9.4, size=2.0, obj_type="obstacle")
    world.place_agent(2.0, 2.0, 0.6)

    return world

def main():
    
    parser = argparse.ArgumentParser(description="Train DQN agent in ContinuousSpace environment.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--steps", type=int, default=200, help="Max steps per episode.")
    args = parser.parse_args()

    state_dim = 6
    action_dim = len(directions)
    agent = DQNAgent(state_dim, action_dim)

    for episode in range(args.episodes):
        env = setup_env()
        state = env.get_state_vector()
        total_reward = 0
        agent.epsilon = agent.epsilon_start

        for step in range(args.steps):
            action_idx = agent.select_action(state)
            reward = env.step_with_reward(action_idx, step_size=0.2, sub_step=0.05)
            next_state = env.get_state_vector()
            done = env.is_task_complete()

            agent.store(state, action_idx, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward
            if done:
                break

        print(f"[Episode {episode+1}] Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    
    agent.epsilon = 0.0
    env = setup_env()
    state = env.get_state_vector()
    total_reward = 0
    steps = 0
    while not env.is_task_complete() and steps < args.steps:
        action_idx = agent.select_action(state)
        reward = env.step_with_reward(action_idx, step_size=0.2, sub_step=0.05)
        state = env.get_state_vector()
        total_reward += reward
        steps += 1

    print(f"[Evaluation] Reward: {total_reward:.2f} | Steps: {steps}")

    img = get_grid_image(env)
    plt.imshow(img)
    plt.title("Environment Snapshot")

    for obj in env.objects:
        if obj['type'] == env.objects_map['boundary']:
            continue

        x, y = obj['x'], obj['y']
        size = obj['size']
        obj_type = obj['type']

        px = int(x / env.width * img.shape[1])
        py = int((env.height - y) / env.height * img.shape[0])

        label_map = {
            env.objects_map['obstacle']: 'Obstacle',
            env.objects_map['target']: 'Target'
        }
        label = label_map.get(obj_type, 'Unknown')

        plt.text(px, py - 15, label,
                 color='black',
                 fontsize=10,
                 ha='center',
                 va='bottom',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    if env.agent is not None:
        ax, ay = env.agent[0]
        px = int(ax / env.width * img.shape[1])
        py = int((env.height - ay) / env.height * img.shape[0])
        plt.text(px, py - 15, 'Agent',
                 color='black',
                 fontsize=10,
                 ha='center',
                 va='bottom',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()

