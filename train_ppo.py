import argparse
from agents.ppo_agent import PPOAgent
import numpy as np
from helper import eval_agent, setup_env, setup_env_hard
import matplotlib.pyplot as plt
import os
from img_gen import get_grid_image

directions = [
    "up",
    "down",
    "left",
    "right",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
]


def plot_and_save_rewards(rewards, save_path):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Average Reward per Episode")
    plt.savefig(save_path)
    plt.close()


def plot_and_save_success_rate(success_history, window, save_path):
    rates = [
        np.mean(success_history[max(0, i - window) : i + 1])
        for i in range(len(success_history))
    ]
    plt.figure()
    plt.plot(rates)
    plt.xlabel("Episode")
    plt.ylabel(f"Success Rate (per {window} episodes)")
    plt.title("Success Rate per Window")
    plt.savefig(save_path)
    plt.close()


def plot_and_save_trajectory(trajectory, save_path):
    trajectory = np.array(trajectory)
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker="o")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Agent Trajectory")
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent in ContinuousSpace environment."
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes."
    )
    parser.add_argument("--steps", type=int, default=300, help="Max steps per episode.")
    parser.add_argument(
        "--sa", action="store_true", help="Enable Strategic Adaptation."
    )
    parser.add_argument("--br", type=int, default=4, help="Number of brackets for SA.")
    parser.add_argument(
        "--tol", type=int, default=100, help="Stopping criterion success history size."
    )
    parser.add_argument(
        "--env",
        type=int,
        default=0,
        help="Which of the 2 env functions to use: 0 for easy, 1 for hard.",
    )
    args = parser.parse_args()

    env = setup_env() if args.env == 0 else setup_env_hard()
    state_dim = len(env.get_state_vector())
    action_dim = len(directions)
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, tol=args.tol)

    success_count = 0
    total_step = 0
    rewards = []
    success_history = []
    completed_flags = []

    save_traj_freq = 20  # Save trajectory every 20 episodes
    update_freq = 5  # Collect data from 5 episodes before each update
    episode_buffer = 0

    for episode in range(1, args.episodes + 1):
        env = setup_env() if args.env == 0 else setup_env_hard()
        state = env.get_state_vector()
        total_reward = 0
        done = False
        step = 0

        while not done and step < args.steps:
            action_idx = agent.take_action(state)
            reward = env.step_with_reward(action_idx, step_size=0.2, sub_step=0.05)
            next_state = env.get_state_vector()
            done = env.is_task_complete()

            agent.store_outcome(reward, done)

            state = next_state
            total_reward += reward
            step += 1

        rewards.append(total_reward)
        completed_flags.append(env.is_task_complete())
        if env.is_task_complete():
            success_count += 1
            total_step += step

        success_history.append(env.is_task_complete())
        if len(success_history) > args.tol:
            success_history.pop(0)
        if args.sa and sum(success_history) == args.tol:
            print(
                f"\nEarly stopping triggered at episode {episode} due to consistent success.\n"
            )
            break

        if args.sa and (episode % (args.episodes // args.br) == 0):
            recent = completed_flags
            if not any(recent):
                print("No success. Restarting agent and continuing.")
                agent = PPOAgent(
                    state_dim=state_dim, action_dim=action_dim, tol=args.tol
                )
                completed_flags = []

        print(
            f"[Episode {episode}] Total reward: {total_reward:.2f} | Success: {env.is_task_complete()}"
        )

        episode_buffer += 1
        if episode_buffer >= update_freq:
            agent.update()  # Update after collecting data from multiple episodes
            episode_buffer = 0

        # Save trajectory
        if episode % save_traj_freq == 0:
            traj_dir = "trajectories"
            os.makedirs(traj_dir, exist_ok=True)
            if hasattr(env, "path"):
                img_path = os.path.join(traj_dir, f"env_path_{episode}.png")
                plt.imshow(get_grid_image(env))
                plt.title("Agent Trajectory - Episode {}".format(episode))
                plt.axis("off")
                plt.savefig(img_path)
                plt.close()

    # Final update for any remaining data
    if episode_buffer > 0:
        agent.update()

    avg_step = total_step / success_count if success_count > 0 else 0.0
    success_rate = success_count / args.episodes
    eval_agent(
        env,
        agent,
        args,
        avg_q_values=np.zeros(args.episodes),
        success_rate=success_rate,
        avg_step=avg_step,
    )

    # Save reward and success rate plots
    plot_and_save_rewards(rewards, "ppo_rewards.png")
    plot_and_save_success_rate(
        success_history, window=10, save_path="ppo_success_rate.png"
    )


if __name__ == "__main__":
    main()
