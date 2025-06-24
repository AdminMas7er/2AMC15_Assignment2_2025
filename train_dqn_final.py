from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
try:
    from world.enviroment_final import ContinuousEnvironment
    from world.continuous_space import ContinuousSpace
except ModuleNotFoundError:
    import sys
    from os import path, pardir
    root_path = path.abspath(path.join(path.join(path.abspath(__file__), pardir), pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world.enviroment_final import ContinuousEnvironment
    from world.continuous_space import ContinuousSpace
from agents.dqn_final import DQNAgent
def parse_args():
    parser=ArgumentParser(description="Train agents in a Continuous Restaurant Environment")
    parser.add_argument("--agent", type=str, default="dqn_final", 
                        help="Name of the agent module (default: random_agent)")
    parser.add_argument("--restaurant", type=Path, default=Path("grid_configs/my_first_restaurant.npz"),
                        help="Path to the .npz restaurant layout (default: my_first_restaurant.npz)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load trained model (optional)")
    parser.add_argument("--demo_mode", action="store_true", help="Run in demo mode (epsilon = 0.0)")
    return parser.parse_args()
def main():
    args=parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    env=ContinuousEnvironment(space_file=args.restaurant, enable_gui=not args.no_gui, seed=args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent=DQNAgent(state_size=env.get_state_size(), action_size=env.get_action_size(),
                   target_update_freq=4, gamma=0.99, batch_size=128,
                   epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                   buffer_size=10000, learning_rate=1e-4,device=device)
    if args.load_model:
        agent.load(args.load_model)
    if args.demo_mode:
        agent.epsilon = 0.0
    observation=env.reset()
    episode_num = 0
    episode_steps = 0
    episode_reward = 0.0
    max_episode_steps = 1000

    episode_rewards = []
    episode_lengths = []
    with trange(args.episodes, desc="Training Episodes") as pbar:
        while episode_num < args.episodes:
            if episode_num <=15:
                max_episode_steps=5000
            else:
                max_episode_steps=1000
            current=observation
            action = agent.action(current, epsilon=agent.epsilon)
            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            episode_reward += reward
            if episode_steps >= max_episode_steps:
                done = True
            agent.observe(current, action, reward, next_state, done)
            observation = next_state
            if next_state is None:
                print("next state is None, skipping the state update")
                continue
            if done:
                pbar.update(1)
                pbar.set_postfix(reward=f"{episode_reward:.2f}", steps=episode_steps)
                episode_num += 1
                observation = env.reset()
                print("episode reward after reset:", episode_reward)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
                episode_steps = 0
                episode_reward = 0.0
    agent.save("trained_model.pth")
    no_deliveries = env.get_no_deliveries()
    print(f"Training completed after {episode_num} episodes.")
    print(f"Total deliveries made: {no_deliveries}")
    print("Model saved to trained_model.pth")

    # Plot learning curves
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(episode_rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1,2,2)
    plt.plot(episode_lengths)
    plt.title("Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.show()

    env.close()
if __name__ == "__main__":
    main()
            
        

