from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
import importlib
import random
import torch
try:
    from world.environment_stefan import ContinuousEnvironment
    from world.continuous_space import ContinuousSpace
except ModuleNotFoundError:
    import sys
    from os import path, pardir
    root_path = path.abspath(path.join(path.join(path.abspath(__file__), pardir), pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world.environment_stefan import ContinuousEnvironment
    from world.continuous_space import ContinuousSpace
from agents.dqn_final import DQNAgent
def parse_args():
    parser=ArgumentParser(description="Train agents in a Continuous Restaurant Environment")
    parser.add_argument("--agent", type=str, default="dqn_final", 
                        help="Name of the agent module (default: random_agent)")
    parser.add_argument("--restaurant", type=Path, default=Path("grid_configs/my_first_restaurant.npz"),
                        help="Path to the .npz restaurant layout (default: my_first_restaurant.npz)")
    parser.add_argument("--iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()
def main():
    args=parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    env=ContinuousEnvironment(space_file=args.restaurant, enable_gui=not args.no_gui, seed=args.seed)
    agent=DQNAgent(state_size=env.get_state_size(), action_size=env.get_action_size(),
                   target_update_freq=1000, gamma=0.99, batch_size=64,
                   epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                   buffer_size=10000, learning_rate=0.00025, episodes=args.iter)
    observation=env.reset()
    for _ in trange(args.iter):
        state=env.reset()
        if observation is None:
            observation=env._get_observation()
            continue
        done=False
        action=agent.action(state, agent.epsilon)
        next_state, reward, done,_ = env.step(action)
        if next_state is None:
            agent.observe(state, action, reward, next_state, done)
        observation=next_state
        if done:
            observation=env.reset()
    env.close()
if __name__ == "__main__":
    main()
            
        

