import numpy as np
from tqdm import trange
import torch
from agents.ppo_agent import PPOAgent
from new_environment import ContinuousSpace
import argparse
import matplotlib.pyplot as plt
from img_gen import get_grid_image

# Hyperparameters
EPISODES = 1000
MAX_STEPS = 200
UPDATE_TIMESTEP = 2048
BATCH_SIZE = 64
STATE_DIM = 10
ACTION_DIM = 8


def create_environment():
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
    return world


def initialize_memory():
    return {
        "states": [],
        "actions": [],
        "logprobs": [],
        "rewards": [],
        "dones": [],
        "values": [],
    }


def collect_experience(env, agent, memory, max_steps, update_timestep, timestep):
    state = env.get_state_vector()
    ep_reward = 0
    for _ in range(max_steps):
        action, logprob, value = agent.select_action(state)
        reward = env.step_with_reward(action)
        next_state = env.get_state_vector()
        done = env.is_task_complete()

        memory["states"].append(state)
        memory["actions"].append(action)
        memory["logprobs"].append(logprob)
        memory["rewards"].append(reward)
        memory["dones"].append(done)
        memory["values"].append(value)

        state = next_state
        ep_reward += reward
        timestep += 1

        if done or timestep % update_timestep == 0:
            break

    return ep_reward, state, timestep, done


@torch.no_grad()
def update_agent(agent, memory, state):
    next_value = agent.model(torch.FloatTensor(state).unsqueeze(0).to(agent.device))[
        1
    ].item()
    returns = agent.compute_gae(
        memory["rewards"], memory["values"], memory["dones"], next_value
    )
    advantages = np.array(returns) - np.array(memory["values"])
    memory["returns"] = returns
    memory["advantages"] = advantages.tolist()
    agent.ppo_update(memory, BATCH_SIZE)
    for k in memory:
        memory[k] = []


def evaluate(agent, steps):
    max_rew = -float("inf")
    best_env = None
    for _ in range(1):
        env = create_environment()
        state = env.get_state_vector()
        total_reward = 0
        step = 0
        while not env.is_task_complete() and step < steps:
            action, _, _ = agent.select_action(state)
            reward = env.step_with_reward(action)
            state = env.get_state_vector()
            total_reward += reward
            step += 1
        print(f"Evaluation Reward: {total_reward:.2f} | Steps: {step}")
        if total_reward > max_rew:
            max_rew = total_reward
            best_env = env
    
    if best_env is not None:
        plt.imshow(get_grid_image(best_env))
        plt.title("Final Path After PPO Training")
        plt.axis("off")
        plt.savefig("ppo_final_path.png", bbox_inches="tight")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent in ContinuousSpace environment."
    )
    parser.add_argument(
        "--episodes", type=int, default=EPISODES, help="Number of training episodes."
    )
    parser.add_argument(
        "--steps", type=int, default=MAX_STEPS, help="Max steps per episode."
    )
    args = parser.parse_args()

    agent = PPOAgent(STATE_DIM, ACTION_DIM)
    all_rewards = []
    memory = initialize_memory()
    timestep = 0

    for ep in trange(args.episodes, desc="Training"):
        env = create_environment()
        ep_reward, last_state, timestep, done = collect_experience(
            env, agent, memory, args.steps, UPDATE_TIMESTEP, timestep
        )
        update_agent(agent, memory, last_state)
        all_rewards.append(ep_reward)
        # if (ep + 1) % 10 == 0:
        #     avg = np.mean(all_rewards[-10:])
        #     print(f"Episode {ep+1}, avg reward: {avg:.2f}")

    print("Training finished.")
    evaluate(agent, args.steps)

    plt.figure()
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Rewards")
    plt.savefig("ppo_training_rewards.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
