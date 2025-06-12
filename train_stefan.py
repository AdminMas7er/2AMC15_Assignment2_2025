"""
Train file for continuous delivery environment using DQN Agent
"""
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
import torch # Added import
import math # Added import

# Ensure DQN_stefan can be imported
try:
    from agents.DQN_stefan import DQNAgent
    from world.environment_stefan import ContinuousEnvironment
    from world.continuous_space import ContinuousSpace
except ModuleNotFoundError:
    import sys
    from os import path, pardir
    root_path = path.abspath(path.join(path.join(path.abspath(__file__), pardir), pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from agents.DQN_stefan import DQNAgent
    from world.environment_stefan import ContinuousEnvironment
    from world.continuous_space import ContinuousSpace

# Define discrete actions: (velocity, rotation_angle_change)
# You can customize these actions
DISCRETE_ACTIONS = [
    (0.1, 0.0),          # Move forward
    (0.0, np.pi / 18),   # Turn left (10 degrees)
    (0.0, -np.pi / 18),  # Turn right (10 degrees)
    (0.05, 0.0),         # Move forward slowly
    # (0.0, 0.0)           # Do nothing (optional)
]
ACTION_SIZE = len(DISCRETE_ACTIONS)

# Max steps per episode
MAX_STEPS_PER_EPISODE = 500 # You can adjust this

def preprocess_observation(obs_dict, env_width, env_height, max_sensor_range):
    """Converts observation dictionary to a flat numpy array for the DQN."""
    agent_pos = obs_dict["agent_pos"]
    agent_angle = obs_dict["agent_angle"]
    sensor_distances = obs_dict["sensor_distances"]
    has_order = obs_dict["has_order"]

    # Normalize sensor distances
    norm_sensor_distances = np.array(sensor_distances) / max_sensor_range

    # Determine target and calculate relative position
    if has_order:
        target_pos = obs_dict["current_target_table"]
    else:
        target_pos = obs_dict["pickup_point"]
    
    if target_pos is None: # Should ideally not happen if logic is correct
        relative_target_pos = np.array([0.0, 0.0])
    else:
        relative_target_pos = target_pos - agent_pos
    
    # Normalize relative target position (approximate, can be improved)
    # Consider using distance and angle to target instead of normalized x, y for better generalization
    norm_relative_target_x = relative_target_pos[0] / env_width
    norm_relative_target_y = relative_target_pos[1] / env_height
    
    # Agent angle components (cyclical feature)
    cos_angle = np.cos(agent_angle)
    sin_angle = np.sin(agent_angle)
    
    has_order_feature = 1.0 if has_order else 0.0
    
    state = np.concatenate([
        norm_sensor_distances,
        [norm_relative_target_x, norm_relative_target_y],
        [has_order_feature],
        [cos_angle, sin_angle]
    ]).astype(np.float32)
    return state

STATE_SIZE = 3 + 2 + 1 + 2 # sensor_dist (3), rel_target_pos (2), has_order (1), angle_components (2)

def parse_args():
    parser = ArgumentParser(description="Train DQN agent in a continuous restaurant space")
    # parser.add_argument("--agent", type=str, default="DQN_stefan", 
    #                     help="Name of the agent module (default: DQN_stefan)") # Kept for consistency, but we directly use DQNAgent
    parser.add_argument("--restaurant", type=Path, default=Path("grid_configs/my_first_restaurant.npz"),
                        help="Path to the .npz restaurant layout (default: my_first_restaurant.npz)")
    parser.add_argument("--iter", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # DQN Hyperparameters (can be moved to DQNAgent defaults or exposed as args)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--target_update", type=int, default=1000, help="Frequency of target network update (in steps)")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate per episode")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # random.seed(args.seed) # Python's random, if used by env or agent's random choices

    # Load environment
    env = ContinuousEnvironment(space_file=args.restaurant, enable_gui=not args.no_gui, seed=args.seed)
    # The environment's max_sensor_range is hardcoded to 5.0 in its _get_sensor_distance method
    # If this changes, the preprocess_observation function needs to be updated or env needs to expose it.
    env_max_sensor_range = 5.0 

    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        target_update_freq=args.target_update,
        gamma=args.gamma,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay, # Agent's internal decay if its train method was used. We manage epsilon externally.
        buffer_size=args.buffer_size,
        learning_rate=args.lr
        # episodes parameter in DQNAgent is not used here, args.iter controls episodes
    )
    # Ensure agent.epsilon is accessible and updatable
    # If DQNAgent.__init__ sets self.epsilon, self.epsilon_end, self.epsilon_decay, we can use them.
    # For this script, we'll manage epsilon decay explicitly per episode.
    current_epsilon = args.epsilon_start


    print(f"Starting training for {args.iter} episodes...")
    print(f"State size: {STATE_SIZE}, Action size: {ACTION_SIZE}")
    
    total_steps = 0
    episode_rewards = []

    for episode in trange(args.iter):
        obs_dict = env.reset()
        current_state = preprocess_observation(obs_dict, env.width, env.height, env_max_sensor_range)
        
        ep_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Select action using agent's action method (assuming it's fixed)
            # The agent.action method should handle the epsilon-greedy logic
            action_idx = agent.action(current_state, current_epsilon)
            env_action = DISCRETE_ACTIONS[action_idx]
            
            next_obs_dict, reward, done = env.step(env_action) # Changed from: next_obs_dict, reward, done, _
            next_state = preprocess_observation(next_obs_dict, env.width, env.height, env_max_sensor_range)
            
            # Store experience in replay buffer
            # Ensure states stored are the preprocessed numpy arrays
            agent.replay_buffer.push((current_state, action_idx, reward, next_state, float(done)))
            
            # Optimize the model
            agent.optimize() # Call the fixed optimize method
            
            current_state = next_state
            ep_reward += reward
            total_steps += 1
            
            # Update target network
            if total_steps % agent.target_update_freq == 0:
                agent.target_network.load_state_dict(agent.q_network.state_dict())
                # print(f"Episode {episode+1}, Step {total_steps}: Target network updated.")

            if done:
                break
        
        episode_rewards.append(ep_reward)
        
        # Decay epsilon after each episode
        current_epsilon = max(args.epsilon_end, current_epsilon * args.epsilon_decay)
        
        if (episode + 1) % 10 == 0: # Log every 10 episodes
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{args.iter} | Avg Reward (last 10): {avg_reward:.2f} | Epsilon: {current_epsilon:.3f} | Total Steps: {total_steps}")

    print("Training finished.")
    env.close()

    # You might want to save the trained model
    # torch.save(agent.q_network.state_dict(), "dqn_agent_model.pth")
    # print("Model saved to dqn_agent_model.pth")

    # Plot rewards (optional)
    # import matplotlib.pyplot as plt
    # plt.plot(episode_rewards)
    # plt.title('Episode Rewards')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.show()

if __name__ == "__main__":
    main()
