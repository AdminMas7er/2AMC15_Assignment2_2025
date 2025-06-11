"""
Train file for continuous delivery environment
"""
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
import importlib
import torch
from agents.sac_agent import SACAgent, preprocess_observation
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
    parser.add_argument("--restaurant", type=Path, default=Path("grid_configs/my_first_restaurant.npz"),
                        help="Path to the .npz restaurant layout (default: my_first_restaurant.npz)")
    parser.add_argument("--iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps per iteration")
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for SAC updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for all optimizers")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update interpolation factor")
    parser.add_argument("--alpha", type=float, default=0.2,help="Initial entropy temperature alpha")

    return parser.parse_args()


def main():
    args = parse_args()
        # Dynamically import agent class
    #agent_module = importlib.import_module(f"agents.{args.agent}")
    #AgentClass = getattr(agent_module, "Agent")

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load environment
    env = ContinuousEnvironment(space_file=args.restaurant, enable_gui=not args.no_gui, seed=args.seed)
    
    # Get initial observation to determine observation space
    initial_observation = env.reset()
    state_vector = preprocess_observation(initial_observation)
    state_dimension = state_vector.shape[0]
    action_dimension = len(env.actions)

    # Initialize the SAC agent
    agent = SACAgent(
        state_dimension=state_dimension,
        action_dimension=action_dimension,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha
    )

    print(f"Training started | State dimension: {state_dimension}, Action dimension: {action_dimension}")
    

    for iteration in trange(args.iter):
        observation = env.reset()
        done = False
        total_reward = 0.0
        # 'Environment steps', aka fill the replay buffer
        for step in trange(args.steps):
            action = agent.take_action(observation=observation)
            next_observation, reward, done = env.step(action)
            total_reward += reward
            agent.append_transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done
            )
            observation = next_observation
            if done:
                break
        print(f"Total reward in iteration {iteration}: {total_reward:.2f}")
        # When the replay buffer has enough samples, update the agent
        if agent.buffer.currentSize >= agent.buffer.minimal_experience:
            print("Updating gradients")
            sampled_batch = agent.buffer.sample_batch()
            agent.update(sampled_batch)


    env.close()

if __name__ == "__main__":
    main()
