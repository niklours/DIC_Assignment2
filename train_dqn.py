import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper import setup_env, setup_env_hard, eval_agent
from agents.dqn_agent import DQNAgent
from agents.DuelingDQN_agent import DuelingDQNAgent


directions = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']

def main():
    parser = argparse.ArgumentParser(description="Train DQN agent in ContinuousSpace environment.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode.")
    parser.add_argument("--sa", action="store_true", help="Enable Strategic Adaptation.")
    parser.add_argument("--br", type=int, default=4, help="Number of brackets for SA.")
    parser.add_argument("--tol", type=int, default=300, help="Stopping criterion success history size.")
    parser.add_argument("--env", type=int, default=0,
                        help="Which of the 2 enviroment functions to use 0 for easy, 1 for hard.")
    parser.add_argument("--dueling", action="store_true")


    args = parser.parse_args()

    state_dim = 7
    action_dim = len(directions)
    Agent = DuelingDQNAgent if args.dueling else DQNAgent
    agent = Agent(state_dim, action_dim, tol=args.tol)

    reward_log, epsilon_log, success_log = [], [], []
    completed_flags, avg_q_values = [], []
    success_episodes, avg_step_total = 0, 0

    for episode in range(args.episodes):
        env = setup_env() if args.env == 0 else setup_env_hard()
        state = env.get_state_vector()
        total_reward, q_sum, steps = 0, 0, 0

        for step in range(args.steps):
            action_idx = agent.take_action(state)
            reward = env.step_with_reward(action_idx, step_size=0.2, sub_step=0.05)
            reward = np.clip(reward, -1.0, 1.0)
            next_state = env.get_state_vector()
            done = env.is_task_complete()

            agent.store(state, action_idx, reward, next_state, done)
            agent.train_step()
            agent.decay_epsilon()

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                q_val = agent.policy(state_tensor).max().item()
                q_sum += q_val

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                success_episodes += 1
                avg_step_total += step
                break

        completed_flags.append(done)
        agent.update_success(done)

        reward_log.append(total_reward)
        epsilon_log.append(agent.epsilon)
        success_log.append(int(done))
        avg_q_values.append(q_sum / steps if steps > 0 else 0)

        print(f"[Episode {episode+1}] Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Success: {done}")

        if args.sa and (episode + 1) % (args.episodes // args.br) == 0:
            recent = completed_flags[-args.br:]
            if not any(recent):
                print("No success in bracket, resetting agent.")
                agent = Agent(state_dim, action_dim, tol=args.tol)
                completed_flags = []

        if agent.early_stop:
            print(f"Early stopping at episode {episode+1}")
            break

    final_env = setup_env() if args.env == 0 else setup_env_hard()
    avg_step = avg_step_total / success_episodes if success_episodes > 0 else 0
    eval_agent(final_env, agent, args, avg_q_values, success_episodes / args.episodes, avg_step)

    df = pd.DataFrame({
        'Episode': np.arange(1, len(reward_log) + 1),
        'Reward': reward_log,
        'Epsilon': epsilon_log,
        'Success': success_log
    })
    df.to_csv("logs/training_log.csv", index=False)

    plt.figure()
    plt.plot(df['Episode'], df['Reward'], label='Reward')
    plt.plot(df['Episode'], df['Epsilon'], label='Epsilon')
    plt.plot(df['Episode'], df['Success'], label='Success')
    plt.xlabel("Episode")
    plt.ylabel("Metrics")
    plt.legend()
    plt.title("Training Progress")
    plt.grid(True)
    plt.savefig("logs/training_metrics.png")
    plt.close()


if __name__ == "__main__":
    main()