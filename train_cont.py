import pygame
import random
from img_gen import get_grid_image
from new_environment import ContinuousSpace

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from agents.dqn_agent import DQNAgent
directions = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']
NUM_EPISODES = 100

def setup_env():
    world = ContinuousSpace(width=11.0, height=11.0, wall_size=1.0)

   
    
    world.add_object(5.0, 5.0, 1.0, "target")

    world.add_rectangle_object(3.0, 3.0, 5.2, 5.4, size=2.0, obj_type="obstacle")
    world.add_rectangle_object(7.0, 7.0, 9.2, 9.4, size=2.0, obj_type="obstacle")
    
    world.place_agent(2.0, 2.0, 0.6)

    return world

def main():
    state_dim = 6  
    action_dim = len(directions)
    agent = DQNAgent(state_dim, action_dim)

    for episode in range(NUM_EPISODES):
        env = setup_env()
        state = env.get_state_vector()
        done = False
        total_reward = 0
        agent.epsilon = agent.epsilon_start
        
        

        for step in range(200):
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

        print(f"[Episode {episode+1}] Total reward: {total_reward:.2f}")
    agent.epsilon = 0.0

    env = setup_env()
    state = env.get_state_vector()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 500:
        action_idx = agent.select_action(state)
        reward = env.step_with_reward(action_idx, step_size=0.2, sub_step=0.05)
        next_state = env.get_state_vector()
        state = next_state
        done = env.is_task_complete()
        total_reward += reward
        steps += 1


    
    import matplotlib.pyplot as plt
    plt.imshow(get_grid_image(env))
    plt.title("Environment Snapshot")
    plt.axis("off")
    plt.show()
 
if __name__ == "__main__":
    main()
