import os
import math
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from img_gen import get_grid_image  
from new_environment import ContinuousSpace  

def eval_agent(env, agent, args, max_rew=0.0, best_env=None):
    state = env.get_state_vector()
    total_reward = 0.0
    q_value_total = 0.0
    reward_list = []
    steps = 0

    while not env.is_task_complete() and steps < args.steps:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_vals = agent.model(state_tensor)
            action = q_vals.argmax().item()
            q_value_total += q_vals.max().item()

        reward = env.step_with_reward(action, step_size=0.2, sub_step=0.05)
        reward_list.append(reward)
        total_reward += reward
        state = env.get_state_vector()
        steps += 1

    print(f"[Evaluation] Reward: {total_reward:.2f} | Steps: {steps}")

    if total_reward > max_rew:
        max_rew = total_reward
        best_env = env

    avg_q_value = q_value_total / steps if steps > 0 else 0

    gamma = agent.gamma
    r = 0.0
    for t in reversed(reward_list):
        r = t + gamma * r
    cum_disc_reward = r

    if env.is_task_complete() and len(env.path) > 1:
        path = env.path
        path_length = sum(
            math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            for i in range(len(path) - 1)
        )
        direct = math.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
        efficiency = direct / path_length if path_length > 0 else 0
    else:
        efficiency = 0.0

    print("\n=== Evaluation Metrics ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average Q-value: {avg_q_value:.4f}")
    print(f"Discounted return: {cum_disc_reward:.2f}")
    print(f"Path efficiency: {efficiency:.2f}")
    print(f"Steps taken: {steps}")
    print("==========================")

    plt.imshow(get_grid_image(best_env or env))
    plt.title("Final Path After Training")
    plt.axis("off")
    plt.show()

    metrics = {
        "total_reward": total_reward,
        "average_q_value": avg_q_value,
        "discounted_return": cum_disc_reward,
        "path_efficiency": efficiency,
        "steps_taken": steps
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{args.steps}_{args.episodes}dqn_metrics.csv"
    filepath = os.path.join("logs", filename)
    os.makedirs("logs", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(filepath, index=False)

    return metrics, max_rew, best_env