"""
Three Deep RL Algorithms Comparison Script
Trains and compares DQN, A3C, and SAC algorithms
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
from agents.a3c_agent_fixed import A3CAgent
from agents.sac_agent import SACAgent

def create_environment(restaurant_file=None, enable_gui=False):
    """Create environment instance"""
    if restaurant_file:
        return DeliveryEnvironment(
            space_file=Path(restaurant_file), 
            enable_gui=enable_gui
        )
    else:
        return DeliveryEnvironment(enable_gui=enable_gui)

def train_all_algorithms(env_fn, config):
    """Train all three algorithms"""
    results = {}
    training_times = {}
    agents = {}
    
    print("🚀 Starting Three-Algorithm Training Comparison")
    print("=" * 80)
    
    # Train DQN
    print("🔵 Training DQN...")
    env = env_fn()
    dqn_agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        lr=config.get('lr', 0.001),
        gamma=config.get('gamma', 0.99),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_end=config.get('epsilon_end', 0.01),
        epsilon_decay=config.get('epsilon_decay', 0.995),
        batch_size=config.get('batch_size', 32),
        target_update_freq=config.get('target_update_freq', 1000),
        buffer_size=config.get('buffer_size', 10000)
    )
    
    start_time = time.time()
    dqn_agent.train(env, num_episodes=config.get('num_episodes', 100))
    training_times['DQN'] = time.time() - start_time
    dqn_agent.save_model('models/dqn_model.pth')
    agents['DQN'] = dqn_agent
    env.close()
    print(f"✅ DQN training completed in {training_times['DQN']:.2f} seconds")
    
    # Train A3C
    print("\n🔴 Training A3C...")
    temp_env = env_fn()
    state_size = temp_env.get_state_size()
    action_size = temp_env.get_action_size()
    temp_env.close()
    
    a3c_agent = A3CAgent(
        state_size=state_size,
        action_size=action_size,
        num_workers=config.get('num_workers', 4),
        lr=config.get('lr', 0.001)
    )
    
    start_time = time.time()
    a3c_agent.train(env_fn, max_episodes=config.get('num_episodes', 100))
    training_times['A3C'] = time.time() - start_time
    a3c_agent.save_model('models/a3c_model.pth')
    agents['A3C'] = a3c_agent
    print(f"✅ A3C training completed in {training_times['A3C']:.2f} seconds")
    
    # Train SAC
    print("\n🟢 Training SAC...")
    env = env_fn()
    sac_agent = SACAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        lr=config.get('lr', 3e-4),
        gamma=config.get('gamma', 0.99),
        tau=config.get('tau', 0.005),
        alpha=config.get('alpha', 0.2),
        buffer_size=config.get('buffer_size', 100000),
        batch_size=config.get('batch_size', 256),
        auto_temp=config.get('auto_temp', True)
    )
    
    start_time = time.time()
    sac_agent.train(env, num_episodes=config.get('num_episodes', 100))
    training_times['SAC'] = time.time() - start_time
    sac_agent.save_model('models/sac_model.pth')
    agents['SAC'] = sac_agent
    env.close()
    print(f"✅ SAC training completed in {training_times['SAC']:.2f} seconds")
    
    return agents, training_times

def evaluate_all_algorithms(agents, env_fn, num_episodes=100):
    """Evaluate all three algorithms"""
    print("\n📊 Starting Algorithm Evaluation")
    print("=" * 80)
    
    results = {}
    
    for algo_name, agent in agents.items():
        print(f"Evaluating {algo_name}...")
        
        rewards = []
        lengths = []
        
        for episode in range(num_episodes):
            env = env_fn()
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = agent.act(state, deterministic=True)
                state, reward, done, _ = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            env.close()
        
        results[algo_name] = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'success_rate': np.mean([r > 0 for r in rewards]),
            'all_rewards': rewards,
            'all_lengths': lengths
        }
        
        print(f"  {algo_name} Results:")
        print(f"    Average Reward: {results[algo_name]['avg_reward']:.2f} ± {results[algo_name]['std_reward']:.2f}")
        print(f"    Average Length: {results[algo_name]['avg_length']:.2f}")
        print(f"    Success Rate: {results[algo_name]['success_rate']:.2%}")
        print()
    
    return results

def create_comprehensive_plots(results, training_times):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Deep RL Algorithms Comprehensive Comparison', fontsize=16, fontweight='bold')
    
    colors = {'DQN': 'blue', 'A3C': 'red', 'SAC': 'green'}
    algorithms = ['DQN', 'A3C', 'SAC']
    
    # 1. Reward Distribution Histograms
    for i, algo in enumerate(algorithms):
        axes[0, i].hist(results[algo]['all_rewards'], bins=20, alpha=0.7, 
                       color=colors[algo], edgecolor='black')
        axes[0, i].set_title(f'{algo} Reward Distribution')
        axes[0, i].set_xlabel('Episode Reward')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axvline(results[algo]['avg_reward'], color='black', 
                          linestyle='--', label=f'Mean: {results[algo]["avg_reward"]:.1f}')
        axes[0, i].legend()
    
    # 2. Performance Metrics Comparison
    metrics = ['Avg Reward', 'Success Rate', 'Avg Length']
    metric_values = {
        'Avg Reward': [results[algo]['avg_reward'] for algo in algorithms],
        'Success Rate': [results[algo]['success_rate'] for algo in algorithms],
        'Avg Length': [results[algo]['avg_length'] for algo in algorithms]
    }
    
    for i, metric in enumerate(metrics):
        bars = axes[1, i].bar(algorithms, metric_values[metric], 
                             color=[colors[algo] for algo in algorithms], alpha=0.7)
        axes[1, i].set_title(f'{metric} Comparison')
        axes[1, i].set_ylabel(metric)
        axes[1, i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values[metric]):
            height = bar.get_height()
            if metric == 'Success Rate':
                axes[1, i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.1%}', ha='center', va='bottom')
            else:
                axes[1, i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:.1f}', ha='center', va='bottom')
    
    # 3. Cumulative Distribution Functions
    axes[2, 0].set_title('Cumulative Reward Distribution')
    for algo in algorithms:
        sorted_rewards = np.sort(results[algo]['all_rewards'])
        cdf = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
        axes[2, 0].plot(sorted_rewards, cdf, label=algo, color=colors[algo], linewidth=2)
    axes[2, 0].set_xlabel('Episode Reward')
    axes[2, 0].set_ylabel('Cumulative Probability')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 4. Box Plot Comparison
    axes[2, 1].boxplot([results[algo]['all_rewards'] for algo in algorithms], 
                       labels=algorithms, patch_artist=True)
    axes[2, 1].set_title('Reward Distribution Box Plot')
    axes[2, 1].set_ylabel('Episode Reward')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Color the box plots
    for patch, algo in zip(axes[2, 1].artists, algorithms):
        patch.set_facecolor(colors[algo])
        patch.set_alpha(0.7)
    
    # 5. Training Time Comparison
    times = [training_times[algo] for algo in algorithms]
    bars = axes[2, 2].bar(algorithms, times, color=[colors[algo] for algo in algorithms], alpha=0.7)
    axes[2, 2].set_title('Training Time Comparison')
    axes[2, 2].set_ylabel('Training Time (seconds)')
    axes[2, 2].grid(True, alpha=0.3)
    
    # Add time labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        axes[2, 2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{time:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/three_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_comparison(results, training_times):
    """Print detailed comparison table"""
    print("\n" + "=" * 100)
    print("🏆 COMPREHENSIVE ALGORITHM COMPARISON")
    print("=" * 100)
    
    # Header
    print(f"{'Metric':<20} {'DQN':<20} {'A3C':<20} {'SAC':<20} {'Best Algorithm':<15}")
    print("-" * 100)
    
    # Metrics comparison
    metrics = [
        ('Avg Reward', 'avg_reward', 'higher'),
        ('Std Reward', 'std_reward', 'lower'),
        ('Avg Length', 'avg_length', 'lower'),
        ('Success Rate', 'success_rate', 'higher'),
        ('Training Time', None, 'lower')
    ]
    
    for metric_name, metric_key, better in metrics:
        if metric_key:
            values = {algo: results[algo][metric_key] for algo in ['DQN', 'A3C', 'SAC']}
            if metric_name == 'Success Rate':
                row_values = [f"{values[algo]:.2%}" for algo in ['DQN', 'A3C', 'SAC']]
            else:
                row_values = [f"{values[algo]:.2f}" for algo in ['DQN', 'A3C', 'SAC']]
        else:  # Training time
            values = training_times
            row_values = [f"{values[algo]:.2f}s" for algo in ['DQN', 'A3C', 'SAC']]
        
        # Find best algorithm
        if better == 'higher':
            best_algo = max(values.keys(), key=lambda k: values[k])
        else:
            best_algo = min(values.keys(), key=lambda k: values[k])
        
        print(f"{metric_name:<20} {row_values[0]:<20} {row_values[1]:<20} {row_values[2]:<20} {best_algo:<15}")
    
    print("=" * 100)
    
    # Algorithm ranking
    print("\n🥇 ALGORITHM RANKING:")
    
    # Calculate overall score (weighted combination of metrics)
    scores = {}
    for algo in ['DQN', 'A3C', 'SAC']:
        score = (
            results[algo]['avg_reward'] * 0.4 +           # 40% weight on reward
            results[algo]['success_rate'] * 100 * 0.3 +   # 30% weight on success rate
            (300 - results[algo]['avg_length']) * 0.2 +   # 20% weight on efficiency (inverse length)
            (1000 / training_times[algo]) * 0.1           # 10% weight on training speed
        )
        scores[algo] = score
    
    ranked_algos = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    
    for i, algo in enumerate(ranked_algos, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"  {emoji} {i}. {algo} (Score: {scores[algo]:.2f})")

def save_comprehensive_results(results, training_times, config):
    """Save comprehensive results to JSON"""
    comprehensive_results = {
        'experiment_config': config,
        'training_times': training_times,
        'algorithm_results': {}
    }
    
    for algo in ['DQN', 'A3C', 'SAC']:
        comprehensive_results['algorithm_results'][algo] = {
            'avg_reward': float(results[algo]['avg_reward']),
            'std_reward': float(results[algo]['std_reward']),
            'avg_length': float(results[algo]['avg_length']),
            'success_rate': float(results[algo]['success_rate']),
            'training_time': float(training_times[algo])
        }
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Save JSON results
    with open('results/three_algorithms_comparison.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n📁 Comprehensive results saved to: results/three_algorithms_comparison.json")

def main():
    parser = argparse.ArgumentParser(description="Three Deep RL Algorithms Comparison")
    parser.add_argument("--restaurant", type=str, default="grid_configs/my_first_restaurant.npz",
                        help="Restaurant layout file path")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--eval_episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--no_gui", action="store_true",
                        help="Disable GUI display")
    
    # Filter out Jupyter/Colab specific arguments
    import sys
    filtered_args = []
    skip_next = False
    
    for i, arg in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith('-f') and 'kernel' in arg:
            continue
        if arg == '-f' and i < len(sys.argv) - 1 and 'kernel' in sys.argv[i + 1]:
            skip_next = True
            continue
        filtered_args.append(arg)
    
    args = parser.parse_args(filtered_args)
    
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Configuration parameters
    config = {
        'num_episodes': args.episodes,
        'eval_episodes': args.eval_episodes,
        'lr': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'target_update_freq': 1000,
        'buffer_size': 10000,
        'num_workers': 4,
        'tau': 0.005,
        'alpha': 0.2,
        'auto_temp': True
    }
    
    # Environment creation function
    env_fn = lambda: create_environment(args.restaurant, enable_gui=not args.no_gui)
    
    print("🚀 Starting Three Deep RL Algorithms Comparison")
    print(f"📍 Restaurant layout: {args.restaurant}")
    print(f"🎯 Training episodes: {args.episodes}")
    print(f"🧪 Evaluation episodes: {args.eval_episodes}")
    print(f"🖥️  GUI enabled: {not args.no_gui}")
    
    # Train all algorithms
    agents, training_times = train_all_algorithms(env_fn, config)
    
    # Evaluate all algorithms
    results = evaluate_all_algorithms(agents, env_fn, args.eval_episodes)
    
    # Create plots
    create_comprehensive_plots(results, training_times)
    
    # Print detailed comparison
    print_detailed_comparison(results, training_times)
    
    # Save results
    save_comprehensive_results(results, training_times, config)
    
    print("\n🎉 Three-algorithm comparison completed successfully!")

if __name__ == "__main__":
    main() 