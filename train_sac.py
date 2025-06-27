"""
SAC Training Script for Restaurant Delivery
"""

import argparse
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from tqdm import trange

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

from agents.sac_agent import SACAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC agent in continuous restaurant environment")
    parser.add_argument("--restaurant", type=Path, default=Path("grid_configs/my_first_restaurant.npz"),
                        help="Path to restaurant layout file")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--eval_freq", type=int, default=100, help="Evaluation frequency (episodes)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--save_freq", type=int, default=500, help="Model saving frequency (episodes)")
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI during training")
    parser.add_argument("--render_eval", action="store_true", help="Render during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--load_model", type=str, help="Path to pre-trained model to load")
    parser.add_argument("--table_radius", type=float, default=0.5, help="Table radius in the environment")
    
    # SAC specific hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor") 
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update coefficient")
    parser.add_argument("--alpha", type=float, default=0.5, help="Initial temperature parameter")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--auto_temp", action="store_true", default=True, help="Automatic temperature tuning")
    
    return parser.parse_args()

def setup_directories(output_dir):
    """Create necessary directories for saving results"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"sac_training_{timestamp}"
    
    dirs = {
        'run': run_dir,
        'models': run_dir / 'models',
        'plots': run_dir / 'plots',
        'logs': run_dir / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def save_config(args, dirs):
    """Save training configuration"""
    config = vars(args)
    config_path = dirs['logs'] / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

def run_evaluation(agent, env, num_episodes, max_steps, render=False):
    """Run evaluation episodes"""
    rewards = []
    lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = agent.act(observation, deterministic=True)
            observation, reward, done, info = env.step(int(action))
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        
        if episode_reward > 0:
            success_count += 1
    
    return {
        'rewards': rewards,
        'lengths': lengths,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'success_rate': success_count / num_episodes
    }

def plot_training_progress(agent, eval_data, eval_episodes, dirs):
    """Create and save training progress plots"""
    
    training_rewards = agent.episode_rewards
    training_lengths = agent.episode_lengths
    training_losses = agent.training_losses
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 12))
    
    if training_rewards:
        axes[0, 0].plot(training_rewards)
        axes[0, 0].set_title('Training Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
    
    if training_lengths:
        axes[0, 1].plot(training_lengths)
        axes[0, 1].set_title('Training Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)

    # collission numbers
    if agent.collisions:
        axes[0, 2].plot(agent.collisions)
        axes[0, 2].set_title('Collisions During Training')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Collisions')
        axes[0, 2].grid(True)
    
    if eval_data:
        eval_means = [data['mean_reward'] for data in eval_data]
        eval_stds = [data['std_reward'] for data in eval_data]
        
        axes[0, 3].errorbar(eval_episodes, eval_means, yerr=eval_stds, marker='o')
        axes[0, 3].set_title('Evaluation Performance')
        axes[0, 3].set_xlabel('Episode')
        axes[0, 3].set_ylabel('Average Reward')
        axes[0, 3].grid(True)
    
    if training_losses['actor_loss']:
        axes[1, 0].plot(training_losses['actor_loss'], label='Actor Loss', alpha=0.7)
        axes[1, 0].plot(training_losses['critic1_loss'], label='Critic1 Loss', alpha=0.7)
        axes[1, 0].plot(training_losses['critic2_loss'], label='Critic2 Loss', alpha=0.7)
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    if training_losses['alpha_value']:
        axes[1, 1].plot(training_losses['alpha_value'])
        axes[1, 1].set_title('Temperature Parameter (Alpha)')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Alpha Value')
        axes[1, 1].grid(True)

    # Success rate over time
    if eval_data:
        success_rates = [data['success_rate'] for data in eval_data]
        axes[1, 2].plot(eval_episodes, success_rates, marker='o')
        axes[1, 2].set_title('Success Rate Over Time')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Success Rate')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(dirs['plots'] / 'sac_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_training_log(agent, eval_data, eval_episodes, dirs):
    """Save detailed training logs"""
    log_data = {
        'training_rewards': agent.episode_rewards,
        'training_lengths': agent.episode_lengths,
        'training_losses': agent.training_losses,
        'collisions': agent.collisions,
        'evaluation_data': eval_data,
        'evaluation_episodes': eval_episodes
    }
    
    log_path = dirs['logs'] / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)

def main():
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup directories
    dirs = setup_directories(args.output_dir)
    save_config(args, dirs)
    
    print(f"Training SAC agent")
    print(f"Results will be saved to: {dirs['run']}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = ContinuousEnvironment(enable_gui=not args.no_gui, space_file=args.restaurant,table_radius=args.table_radius)
    
    # Get environment dimensions
    obs = env.reset()
    state_size = 26
    action_size = 5
    
    # Initialize SAC agent
    agent = SACAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device=device,
        hidden_size=args.hidden_size
    )
    
    # Enhanced warmup phase
    print(f"Enhanced warmup phase: collecting {args.batch_size * 4} initial experiences...")
    warmup_steps = args.batch_size * 4
    warmup_obs = env.reset()
    successful_episodes = 0
    
    for step in range(warmup_steps):
        if step < warmup_steps // 2:
            warmup_action = np.random.randint(0, 5)
        else:
            if np.random.random() < 0.4:
                warmup_action = 3
            else:
                warmup_action = np.random.randint(0, 5)
                
        next_obs, reward, done, info = env.step(warmup_action)
        agent.remember(warmup_obs, warmup_action, reward, next_obs, done)
        
        if done:
            if reward > 40:
                successful_episodes += 1
            warmup_obs = env.reset()
        else:
            warmup_obs = next_obs
            
    print(f"Enhanced warmup completed! Buffer size: {len(agent.memory)}, Successful episodes: {successful_episodes}")
    
    # Load pre-trained model if specified
    if args.load_model:
        agent.load_model(args.load_model)
        print(f"Loaded pre-trained model from: {args.load_model}")
    
    # Training variables
    eval_data = []
    eval_episodes = []
    best_reward = float('-inf')
    
    print(f"Starting SAC training for {args.episodes} episodes...")
    print(f"Evaluation every {args.eval_freq} episodes")
    
    # Training loop
    for episode in trange(args.episodes, desc="Training SAC"):
        observation = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_collisions = 0
        
        for step in range(args.max_steps):
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(int(action))
            
            # Track collisions from the info dict
            if info.get('collision', False):
                episode_collisions += 1
            
            agent.remember(observation, action, reward, next_observation, done)
            
            # Delayed training
            min_experiences = args.batch_size * 8
            if len(agent.memory) > min_experiences:
                if episode < 25:
                    update_freq = 1
                elif episode < 50:
                    update_freq = 2
                else:
                    update_freq = 2
                    
                for _ in range(update_freq):
                    agent.update_networks()
            
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)
        agent.collisions.append(episode_collisions)
        
        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            print(f"\nEvaluating at episode {episode + 1}...")
            eval_results = run_evaluation(agent, env, args.eval_episodes, args.max_steps, render=args.render_eval)
            eval_data.append(eval_results)
            eval_episodes.append(episode + 1)
            
            # Current temperature value
            current_alpha = agent.log_alpha.exp().item() if agent.auto_temp else agent.alpha
            
            print(f" Evaluation Results (Episode {episode + 1}):")
            print(f"   Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"   Mean Length: {eval_results['mean_length']:.1f}")
            print(f"   Success Rate: {eval_results['success_rate']:.2%}")
            print(f"   Temperature (α): {current_alpha:.4f}")
            print(f"   Buffer Size: {len(agent.memory)}")
            print("-" * 60)
            
            # Save best model
            if eval_results['mean_reward'] > best_reward:
                best_reward = eval_results['mean_reward']
                best_model_path = dirs['models'] / 'best_model.pth'
                agent.save_model(best_model_path)
                print(f"New best model saved! Reward: {best_reward:.2f}")
    
    print(f"\n Running final evaluation...")
    
    # Final evaluation
    final_eval = run_evaluation(
        agent, env, args.eval_episodes * 2, args.max_steps, 
        render=args.render_eval
    )
    
    print(f"\n Final SAC Evaluation Results:")
    print(f"   Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"   Mean Length: {final_eval['mean_length']:.1f}")
    print(f"   Success Rate: {final_eval['success_rate']:.2%}")
    
    # Save final model
    final_model_path = dirs['models'] / 'final_model.pth'
    agent.save_model(final_model_path)
    print(f" Final model saved: {final_model_path}")
    
    # Calculate collision statistics for the final report
    total_collisions = sum(agent.collisions) if agent.collisions else 0
    
    # Save final results
    final_results = {
        'final_evaluation': final_eval,
        'best_evaluation_reward': best_reward,
        'total_episodes': args.episodes,
        'total_training_collisions': total_collisions,
        'hyperparameters': vars(args)
    }
    
    results_path = dirs['logs'] / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Generate plots and save logs
    plot_training_progress(agent, eval_data, eval_episodes, dirs)
    save_training_log(agent, eval_data, eval_episodes, dirs)
    
    print(f"\n SAC training completed!")
    print(f" All results saved to: {dirs['run']}")

if __name__ == "__main__":
    main() 