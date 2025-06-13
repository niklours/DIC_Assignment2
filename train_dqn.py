import argparse
import matplotlib.pyplot as plt
from img_gen import get_grid_image
from new_environment import ContinuousSpace  
from agents.dqn_agent import DQNAgent  
import sys
import os
import numpy as np
directions = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']

def setup_env():
    world = ContinuousSpace(width=11.0, height=11.0, wall_size=1.0)
    world.add_object(5.0, 5.0, 1.0, "target")
    obstacle_coords = [
        (3.0, 3.0, 3.2, 3.2),
        (3.0, 2.3, 3.2, 2.4),
        (7.0, 7.0, 8.0, 8.0),     
        (2.0, 6.0, 3.5, 7.0),     
    ]
    for x1, y1, x2, y2 in obstacle_coords:
        world.add_rectangle_object(x1, y1, x2, y2, size=1.0, obj_type="obstacle")


    world.place_agent(2.0, 2.0, 0.6)

    world.place_agent(2.0, 2.0, 0.6)
    return world

def main():
    parser = argparse.ArgumentParser(description="Train DQN agent in ContinuousSpace environment.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode.")
    args = parser.parse_args()

    state_dim = 10
    action_dim = len(directions)
    agent = DQNAgent(state_dim, action_dim)

    for episode in range(args.episodes):
        env = setup_env()
        state = env.get_state_vector()
        total_reward = 0
         
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

    max_rew=-(np.inf)
    
    for _ in range(1):
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
        if total_reward > max_rew:
            max_rew = total_reward
            best_env = env
    plt.imshow(get_grid_image(best_env))
    plt.title("Final Path After Training")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()

