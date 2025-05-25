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
    parser.add_argument("--agent", type=str, required=True, help="Name of the agent module (e.g., random_agent)")
    parser.add_argument("--restaurant", type=Path, required=True, help="Path to .npz restaurant layout")
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
    agent = AgentClass()

    observation = env.reset()

    for _ in trange(args.iter):
        action = agent.take_action(observation)
        observation = env.step(action)

    env.close()

if __name__ == "__main__":
    main()
