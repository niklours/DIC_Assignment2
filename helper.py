import os
import math
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from img_gen import get_grid_image  
from new_environment import ContinuousSpace  

import numpy as np
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

    return world

def setup_env_hard():
    world = ContinuousSpace(width=11.0, height=11.0, wall_size=1.0)
    world.add_object(8.2, 8.2, 1.0, "target")
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


def eval_agent(env, agent, args,avg_q_values):
    state = env.get_state_vector()
    done = False
    total_reward = 0
    steps = 0
    q_value_total = 0.0
    reward_list = []

    while not done and steps < args.steps:
        action_idx = agent.select_action(state, deterministic=True)
        reward = env.step_with_reward(action_idx, step_size=0.2, sub_step=0.05)
        state = env.get_state_vector()
        done = env.is_task_complete()
        total_reward += reward
        reward_list.append(reward)
        steps += 1

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            q_val = agent.model(state_tensor).max().item()
            q_value_total += q_val

    avg_q_value = q_value_total / steps if steps > 0 else 0.0
    gamma = agent.gamma
    discounted_return = sum(gamma**t * r for t, r in enumerate(reward_list))

    if env.is_task_complete() and len(env.path) > 1:
        path = env.path
        path_length = sum(
            math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            for i in range(len(path) - 1)
        )
        direct = math.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
        efficiency = direct / path_length if path_length > 0 else 0.0
    else:
        efficiency = 0.0

    print("\n=== Evaluation Metrics ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average Q-value: {avg_q_value:.4f}")
    print(f"Discounted return: {discounted_return:.2f}")
    print(f"Path efficiency: {efficiency:.2f}")
    print(f"Steps taken: {steps}")
    print("==========================")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.env == 0:
        env_name = "easy"
    else:
        env_name = "hard"

    if args.sa:
        sa=args.tol
    else:
        sa = "no_sa"
    folder_name = f"{timestamp}_{args.steps}_{args.episodes}_{sa}_{args.br}_{env_name}"
    log_dir = os.path.join("logs", folder_name)
    os.makedirs(log_dir, exist_ok=True)

    img_path = os.path.join(log_dir, "env_path.png")
    plt.imshow(get_grid_image(env))
    plt.title("Final Path After Training")
    plt.axis("off")
    plt.savefig(img_path)
    plt.close()

    metrics = {
        "total_reward": total_reward,
        "average_q_value": avg_q_value,
        "discounted_return": discounted_return,
        "path_efficiency": efficiency,
        "steps_taken": steps
    }
    q_vals = np.array(avg_q_values)
    q_norm = (q_vals - q_vals.min()) / (q_vals.max() - q_vals.min() + 1e-8)

    plt.figure(figsize=(8, 4))
    plt.plot(q_norm, label="Normalized Avg Q-value", color="blue", marker='o', linestyle='-', markersize=3)
    plt.title("Normalized Q-value per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Q-value (Normalized)")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", folder_name)
    os.makedirs(log_dir, exist_ok=True)

    q_plot_path = os.path.join(log_dir, "q_convergence_plot.png")
    plt.savefig(q_plot_path)
    plt.close()
    csv_path = os.path.join(log_dir, "dqn_metrics.csv")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)