import argparse
import matplotlib.pyplot as plt
from new_environment import ContinuousSpace
from agents.ppo_agent import PPOAgent
import torch
import numpy as np
import pandas as pd
from helper import eval_agent, setup_env, setup_env_hard

directions = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent in ContinuousSpace environment.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode.")
    parser.add_argument("--sa", action="store_true", help="Enable Strategic Adaptation.")
    parser.add_argument("--br", type=int, default=4, help="Number of brackets for SA.")
    parser.add_argument("--tol", type=int, default=300, help="Stopping criterion success history size.")
    parser.add_argument("--env", type=int, default=0, help="Which of the 2 env functions to use: 0 for easy, 1 for hard.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    args = parser.parse_args()

    env = setup_env() if args.env == 0 else setup_env_hard()
    state_dim = len(env.get_state_vector())
    action_dim = len(directions)
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, tol=args.tol, lr=args.lr, gamma=args.gamma)

    # 熵系数衰减参数
    init_entropy_coef = 0.9
    final_entropy_coef = 0.2
    decay_episodes = args.episodes // 2

    success_count = 0
    total_step = 0
    rewards = []
    q_values = []

    for episode in range(1, args.episodes + 1):
        env = setup_env() if args.env == 0 else setup_env_hard()
        state = env.get_state_vector()
        total_reward = 0
        done = False
        step = 0

        # --- 动态调整熵系数 ---
        if episode <= decay_episodes:
            agent.entropy_coef = init_entropy_coef - (init_entropy_coef - final_entropy_coef) * (episode / decay_episodes)
        else:
            agent.entropy_coef = final_entropy_coef
        # --- end ---

        while not done and step < args.steps:
            action_idx = agent.take_action(state)
            reward = env.step_with_reward(action_idx, step_size=0.2, sub_step=0.05)
            next_state = env.get_state_vector()
            done = env.is_task_complete()

            agent.store_outcome(reward, done)
            state = next_state
            total_reward += reward
            step += 1
            
        pos = (state[0] * env.width, state[1] * env.height)
        print(f'agent position: {[round(coord, 2) for coord in pos]}')
        

        agent.update()
        rewards.append(total_reward)

        with torch.no_grad():
            _, value = agent.actor_critic(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            q_values.append(value.item())

        if env.is_task_complete():
            success_count += 1
            total_step += step

            print(f"[Episode {episode}] Total reward: {total_reward:.2f} | Success: {env.is_task_complete()}")

    avg_step = total_step / success_count if success_count > 0 else 0.0
    success_rate = success_count / args.episodes

    print(f"\n[Summary] Success Rate: {success_rate:.2f}, Avg Steps: {avg_step:.2f}, Avg Q: {np.mean(q_values):.2f}")

    df = pd.DataFrame({
        "episode": list(range(1, args.episodes + 1)),
        "reward": rewards,
        "q_value": q_values
    })
    df.to_csv("ppo_training_log.csv", index=False)

if __name__ == "__main__":
    main()
