"""
Train file for continuous delivery environment
"""
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
import importlib

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
    return parser.parse_args()


def main():
    args = parse_args()

    # Dynamically import agent class
    agent_module = importlib.import_module(f"agents.{args.agent}")
    AgentClass = getattr(agent_module, "Agent")

    # Load environment
    env = ContinuousEnvironment(space_file=args.restaurant, enable_gui=not args.no_gui)
    agent = AgentClass(state_size=env.get_state_size(), action_size=env.get_action_size())

    observation = env.reset()

    for _ in trange(args.iter):
        if observation is None:
            # Skip bad observation
            observation = env.reset()
            continue

        action = agent.take_action(observation)
        next_state, reward, done, _ = env.step(action)

        if next_state is not None:
            agent.observe_transition(observation, action, reward, next_state, done)

        observation = next_state

        if done:
            observation = env.reset()

    env.close()

if __name__ == "__main__":
    main()
