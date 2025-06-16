import argparse
import torch
from new_environment import ContinuousSpace
from agents.dueldqn import DuelingDQNAgent
from plot_path import plot_path
from img_gen import get_grid_image
import matplotlib.pyplot as plt
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps",    type=int, default=100)
    parser.add_argument("--width",    type=float, default=10.0)
    parser.add_argument("--height",   type=float, default=10.0)
    args = parser.parse_args()

    env = ContinuousSpace(args.width, args.height)
    env.place_agent(x=1.0, y=1.0, size=0.5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DuelingDQNAgent(
        state_dim=10,
        action_dim=8,
        device=device
    )

    for ep in range(args.episodes):
        env = ContinuousSpace(args.width, args.height)
        env.place_agent(x=1.0, y=1.0, size=0.5)

        state = env.get_state_vector()
        #print("state vector length:", len(state))

        total_reward = 0.0

        for t in range(args.steps):
            action = agent.select_action(state)
            reward = env.step_with_reward(action)
            next_state = env.get_state_vector()
            done = env.is_task_complete()

            agent.store(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
            if done:
                break

        print(f"Episode {ep}  reward={total_reward:.1f}")

    agent.eps_start = agent.eps_end = 0.0  # force greedy
    env = ContinuousSpace(args.width, args.height)
    env.place_agent(x=1.0, y=1.0, size=0.5)
    state = env.get_state_vector()
    for _ in range(args.steps):
        a = agent.select_action(state)
        env.step_with_reward(a)
        state = env.get_state_vector()
        if env.is_task_complete():
            break

    #plot_path(env, filename=f"episode_{ep}_path.png")
    img = get_grid_image(env, resolution=256)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("Agent’s Final Path")
    plt.show()

if __name__ == "__main__":
    main()
