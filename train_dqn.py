import argparse
import matplotlib.pyplot as plt
from img_gen import get_grid_image
from new_environment import ContinuousSpace  
from agents.dqn_agent import DQNAgent  
import sys
import torch
import  math
import os
import numpy as np
from datetime import datetime
import pandas as pd
from helper import eval_agent,setup_env,setup_env_hard

directions = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']



def main():
    parser = argparse.ArgumentParser(description="Train DQN agent in ContinuousSpace environment.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode.")
    parser.add_argument("--sa", action="store_true", help="Enable Strategic Adaptation.")
    parser.add_argument("--br", type=int, default=4, help="Number of brackets for SA.")
    parser.add_argument("--tol",type=int, default=30, help="Stopping criterion sucess history size.")
    parser.add_argument("--env",type=int, default=0, help="Which of the 2 enviroment functions to use 0 for easy, 1 for hard.")

    args = parser.parse_args()

    state_dim = 7
    action_dim = len(directions)
    agent = DQNAgent(state_dim, action_dim, args.tol)
    completed_flags = []
    avg_q_values = []
    for episode in range(args.episodes):
        if args.env == 0:
            env = setup_env()
        else:
            env = setup_env_hard()
        state = env.get_state_vector()
        total_reward = 0
        q_sum = 0.0
        steps = 0

        for step in range(args.steps):
            action_idx = agent.take_action(state)

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                q_val = agent.policy(state_tensor).max().item()
                q_sum += q_val
                steps += 1

            reward = env.step_with_reward(action_idx, step_size=0.2, sub_step=0.05)
            next_state = env.get_state_vector()
            done = env.is_task_complete()

            agent.store(state, action_idx, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        completed_flags.append(done)
        

        agent.update_success(done)

        avg_q_val = q_sum / steps if steps > 0 else 0.0
        avg_q_values.append(avg_q_val)

        print(f"[Episode {episode+1}] Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f} | Success: {done}")

        if args.sa and (episode + 1) % (args.episodes // args.br) == 0:
            recent = completed_flags
            if not any(recent):
                print("No success. Restarting agent and continuing.")
                agent = DQNAgent(state_dim, action_dim,args.tol)  
                completed_flags = []

   
        if agent.early_stop:
            print(f"\nEarly stopping triggered at episode {episode+1} due to consistent success.\n")
            break


    

    if args.env == 0:
        env = setup_env()
    else:
        env = setup_env_hard()
    
    eval_agent(env, agent, args,avg_q_values)


if __name__ == "__main__":
    main()


