"""
Deep Reinforcement Learning Training Script
Supports training, evaluation and comparison of DQN and A3C algorithms
"""

import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import environment and agents
from world.delivery_environment import DeliveryEnvironment
from agents.dqn_agent import DQNAgent
from agents.a3c_agent import A3CAgent

def create_environment(restaurant_file=None, enable_gui=False):
    """Create environment instance"""
    if restaurant_file:
        return DeliveryEnvironment(
            space_file=Path(restaurant_file), 
            enable_gui=enable_gui
        )
    else:
        return DeliveryEnvironment(enable_gui=enable_gui)

def train_dqn(env, config):
    """Train DQN agent"""
    print("=" * 60)
    print("Starting DQN Agent Training")
    print("=" * 60)
    
    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        lr=config.get('lr', 0.00025),
        gamma=config.get('gamma', 0.99),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_end=config.get('epsilon_end', 0.01),
        epsilon_decay=config.get('epsilon_decay', 0.995),
        batch_size=config.get('batch_size', 32),
        target_update_freq=config.get('target_update_freq', 1000),
        buffer_size=config.get('buffer_size', 10000)
    )
    
    # Training
    start_time = time.time()
    agent.train(env, num_episodes=config.get('num_episodes', 2000))
    training_time = time.time() - start_time
    
    # Save model
    agent.save_model('models/dqn_model.pth')
    
    print(f"DQN training time: {training_time:.2f} seconds")
    
    return agent, training_time

def train_a3c(env_fn, config):
    """Train A3C agent"""
    print("=" * 60)
    print("Starting A3C Agent Training")
    print("=" * 60)
    
    # Create sample environment to get state and action space sizes
    temp_env = env_fn()
    state_size = temp_env.get_state_size()
    action_size = temp_env.get_action_size()
    temp_env.close()
    
    agent = A3CAgent(
        state_size=state_size,
        action_size=action_size,
        num_workers=config.get('num_workers', 4),
        lr=config.get('lr', 0.001)
    )
    
    # Training
    start_time = time.time()
    agent.train(env_fn, max_episodes=config.get('num_episodes', 1000))
    training_time = time.time() - start_time
    
    # Save model
    agent.save_model('models/a3c_model.pth')
    
    print(f"A3C training time: {training_time:.2f} seconds")
    
    return agent, training_time

def evaluate_and_compare(dqn_agent, a3c_agent, env_fn, num_episodes=100):
    """Evaluate and compare both agents"""
    print("=" * 60)
    print("Starting Agent Evaluation and Comparison")
    print("=" * 60)
    
    # Evaluate DQN
    print("Evaluating DQN...")
    env = env_fn()
    dqn_results = dqn_agent.evaluate(env, num_episodes)
    env.close()
    
    # Evaluate A3C
    print("\nEvaluating A3C...")
    env = env_fn()
    
    # A3C evaluation
    a3c_rewards = []
    a3c_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = a3c_agent.act(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        a3c_rewards.append(episode_reward)
        a3c_lengths.append(episode_length)
    
    env.close()
    
    a3c_results = {
        'avg_reward': np.mean(a3c_rewards),
        'std_reward': np.std(a3c_rewards),
        'avg_length': np.mean(a3c_lengths),
        'success_rate': np.mean([r > 0 for r in a3c_rewards]),
        'all_rewards': a3c_rewards
    }
    
    print(f"\nA3C Evaluation Results ({num_episodes} episodes):")
    print(f"  Average Reward: {a3c_results['avg_reward']:.2f} ± {a3c_results['std_reward']:.2f}")
    print(f"  Average Length: {a3c_results['avg_length']:.2f}")
    print(f"  Success Rate: {a3c_results['success_rate']:.2%}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("Algorithm Comparison")
    print("=" * 60)
    print(f"{'Metric':<15} {'DQN':<15} {'A3C':<15} {'A3C Advantage':<15}")
    print("-" * 60)
    print(f"{'Avg Reward':<15} {dqn_results['avg_reward']:<15.2f} {a3c_results['avg_reward']:<15.2f} {a3c_results['avg_reward'] - dqn_results['avg_reward']:<15.2f}")
    print(f"{'Avg Length':<15} {dqn_results['avg_length']:<15.2f} {a3c_results['avg_length']:<15.2f} {a3c_results['avg_length'] - dqn_results['avg_length']:<15.2f}")
    print(f"{'Success Rate':<15} {dqn_results['success_rate']:<15.2%} {a3c_results['success_rate']:<15.2%} {a3c_results['success_rate'] - dqn_results['success_rate']:<15.2%}")
    
    return dqn_results, a3c_results

def plot_results(dqn_results, a3c_results):
    """Plot comparison results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward distribution comparison
    axes[0, 0].hist(dqn_results['all_rewards'], bins=20, alpha=0.7, label='DQN', color='blue')
    axes[0, 0].hist(a3c_results['all_rewards'], bins=20, alpha=0.7, label='A3C', color='red')
    axes[0, 0].set_xlabel('Episode Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Reward Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Performance metric comparison
    metrics = ['Avg Reward', 'Success Rate']
    dqn_values = [dqn_results['avg_reward'], dqn_results['success_rate']]
    a3c_values = [a3c_results['avg_reward'], a3c_results['success_rate']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, dqn_values, width, label='DQN', color='blue', alpha=0.7)
    axes[0, 1].bar(x + width/2, a3c_values, width, label='A3C', color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Metrics')
    axes[0, 1].set_title('Performance Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative reward distribution
    dqn_sorted = np.sort(dqn_results['all_rewards'])
    a3c_sorted = np.sort(a3c_results['all_rewards'])
    dqn_cdf = np.arange(1, len(dqn_sorted) + 1) / len(dqn_sorted)
    a3c_cdf = np.arange(1, len(a3c_sorted) + 1) / len(a3c_sorted)
    
    axes[1, 0].plot(dqn_sorted, dqn_cdf, label='DQN', color='blue', linewidth=2)
    axes[1, 0].plot(a3c_sorted, a3c_cdf, label='A3C', color='red', linewidth=2)
    axes[1, 0].set_xlabel('Episode Reward')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Cumulative Distribution Function')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    axes[1, 1].boxplot([dqn_results['all_rewards'], a3c_results['all_rewards']], 
                       labels=['DQN', 'A3C'])
    axes[1, 1].set_ylabel('Episode Reward')
    axes[1, 1].set_title('Reward Distribution (Box Plot)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(dqn_results, a3c_results, dqn_time, a3c_time, config):
    """Save experiment results"""
    results = {
        'experiment_config': config,
        'dqn_results': {
            'avg_reward': float(dqn_results['avg_reward']),
            'std_reward': float(dqn_results['std_reward']),
            'avg_length': float(dqn_results['avg_length']),
            'success_rate': float(dqn_results['success_rate']),
            'training_time': dqn_time
        },
        'a3c_results': {
            'avg_reward': float(a3c_results['avg_reward']),
            'std_reward': float(a3c_results['std_reward']),
            'avg_length': float(a3c_results['avg_length']),
            'success_rate': float(a3c_results['success_rate']),
            'training_time': a3c_time
        }
    }
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Save JSON results
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment results saved to: results/experiment_results.json")

def main():
    parser = argparse.ArgumentParser(description="Deep Reinforcement Learning Training Script")
    parser.add_argument("--algorithm", choices=['dqn', 'a3c', 'both'], default='both',
                        help="Choose training algorithm")
    parser.add_argument("--restaurant", type=str, default="grid_configs/my_first_restaurant.npz",
                        help="Restaurant layout file path")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--eval_episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--no_gui", action="store_true",
                        help="Disable GUI display")
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Configuration parameters
    config = {
        'num_episodes': args.episodes,
        'lr': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'target_update_freq': 1000,
        'buffer_size': 10000,
        'num_workers': 4
    }
    
    # Environment creation function
    env_fn = lambda: create_environment(args.restaurant, enable_gui=not args.no_gui)
    
    print("Starting Deep Reinforcement Learning Experiment")
    print(f"Restaurant layout: {args.restaurant}")
    print(f"Training episodes: {args.episodes}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    
    if args.algorithm in ['dqn', 'both']:
        env = env_fn()
        dqn_agent, dqn_time = train_dqn(env, config)
        env.close()
    
    if args.algorithm in ['a3c', 'both']:
        a3c_agent, a3c_time = train_a3c(env_fn, config)
    
    if args.algorithm == 'both':
        # Compare both algorithms
        dqn_results, a3c_results = evaluate_and_compare(
            dqn_agent, a3c_agent, env_fn, args.eval_episodes
        )
        
        # Plot results
        plot_results(dqn_results, a3c_results)
        
        # Save results
        save_results(dqn_results, a3c_results, dqn_time, a3c_time, config)

if __name__ == "__main__":
    main() 