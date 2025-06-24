"""
Train file for continuous delivery environment
"""
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
import importlib
import matplotlib.pyplot as plt

try:
    from world.environment import ContinuousEnvironment
    from world.continuous_space import ContinuousSpace
except ModuleNotFoundError:
    import sys
    from os import path, pardir
    root_path = path.abspath(path.join(path.join(path.abspath(__file__), pardir), pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world.environment import ContinuousEnvironment
    from world.continuous_space import ContinuousSpace

def parse_args():
    parser = ArgumentParser(description="Train agent in a continuous restaurant space")
    parser.add_argument("--agent", type=str, default="random_agent", 
                        help="Name of the agent module (default: random_agent)")
    parser.add_argument("--restaurant", type=Path, default=Path("grid_configs/my_first_restaurant.npz"),
                        help="Path to the .npz restaurant layout (default: my_first_restaurant.npz)")
    parser.add_argument("--iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load trained model (optional)")
    parser.add_argument("--demo_mode", action="store_true", help="Run in demo mode (epsilon = 0.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Dynamically import agent class
    agent_module = importlib.import_module(f"agents.{args.agent}")
    AgentClass = getattr(agent_module, "Agent")

    # Load environment
    env = ContinuousEnvironment(space_file=args.restaurant, enable_gui=not args.no_gui)
    agent = AgentClass(state_size=env.get_state_size(), action_size=env.get_action_size())

    # load trained model
    if args.load_model is not None:
        agent.load_model(args.load_model)
        print("Running with loaded model.")

    # demo mode (force epsilon = 0.0)
    if args.demo_mode:
        agent.epsilon = 0.0
        print("Running in demo mode (epsilon = 0.0)")

    observation = env.reset()

    # Episode monitor variables
    episode_num = 0
    episode_steps = 0
    episode_reward = 0.0
    max_episode_steps = 1500

    episode_rewards = []
    episode_lengths = []

    for _ in trange(args.iter):
        # if observation is None:
        #     # Skip bad observation
        #     print("No observation, resetted")
        #     observation = env.reset()
        #     continue

        action = agent.take_action(observation)
        next_state, reward, done, _ = env.step(action)

        # Update episode monitor
        episode_steps += 1
        episode_reward += reward

        if episode_steps >= max_episode_steps:
            done=True
            print(f"Max steps reached → forcing episode end.")

        observation = next_state

        if next_state is not None:
            agent.observe_transition(observation, action, reward, next_state, done)

        # observation = next_state

        if done:
            episode_num += 1
            print(f"Episode {episode_num} ended → Steps: {episode_steps}, Total reward: {episode_reward:.2f}")

            # Save stats for plotting
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)

            observation = env.reset()

            # Reset monitor variables
            episode_steps = 0
            episode_reward = 0.0

    # After training → save model
    agent.save_model("trained_model.pth")
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





