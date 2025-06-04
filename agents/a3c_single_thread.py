"""
单线程 A3C 实现 - 避免 Colab 多线程问题
保留A3C的核心算法逻辑，但使用单线程执行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import time

class ActorCriticNetwork(nn.Module):
    """Actor-Critic Network for single-thread A3C"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor_head = nn.Linear(hidden_size, action_size)
        
        # Critic head (value function)
        self.critic_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor output (action probabilities)
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic output (state value)
        value = self.critic_head(x)
        
        return action_probs, value

class SingleThreadA3C:
    """单线程A3C - 保留算法逻辑，避免多线程问题"""
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, entropy_coeff=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        
        # Network
        self.network = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training parameters
        self.update_frequency = 20  # Update every 20 steps
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.losses = deque(maxlen=100)
        
        print(f"Single-thread A3C Agent initialized:")
        print(f"  State size: {state_size}")
        print(f"  Action size: {action_size}")
        print(f"  Learning rate: {lr}")
    
    def compute_gae(self, rewards, values, next_value, dones, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def act(self, state, deterministic=False):
        """Select action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, _ = self.network(state_tensor)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=1).item()
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        
        return action
    
    def train(self, env, num_episodes=1000):
        """Train single-thread A3C"""
        print(f"Starting single-thread A3C training ({num_episodes} episodes)...")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Collect experience
            states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
            
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.update_frequency):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Get action and value
                action_probs, value = self.network(state_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                next_state, reward, done, _ = env.step(action.item())
                
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob.item())
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Calculate next state value
            if done:
                next_value = 0
            else:
                with torch.no_grad():
                    _, next_value = self.network(torch.FloatTensor(state).unsqueeze(0))
                    next_value = next_value.item()
            
            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values, next_value, dones)
            
            # Update network
            loss_info = self.update_network(states, actions, advantages, returns)
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                if loss_info:
                    self.losses.append(loss_info['total_loss'])
                
                # Progress reporting
                if episode % 50 == 0:
                    avg_reward = np.mean(list(self.episode_rewards)[-50:])
                    avg_length = np.mean(list(self.episode_lengths)[-50:])
                    avg_loss = np.mean(list(self.losses)[-50:]) if self.losses else 0
                    
                    print(f"Episode {episode}/{num_episodes}")
                    print(f"  Avg Reward (last 50): {avg_reward:.2f}")
                    print(f"  Avg Length: {avg_length:.2f}")
                    print(f"  Avg Loss: {avg_loss:.4f}")
                    print("-" * 40)
        
        training_time = time.time() - start_time
        print(f"Single-thread A3C training completed in {training_time:.2f} seconds!")
    
    def update_network(self, states, actions, advantages, returns):
        """Update network"""
        try:
            self.optimizer.zero_grad()
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(np.array(states))
            actions_tensor = torch.LongTensor(np.array(actions))
            advantages_tensor = torch.FloatTensor(np.array(advantages))
            returns_tensor = torch.FloatTensor(np.array(returns))
            
            # Forward pass
            action_probs, values = self.network(states_tensor)
            
            # Ensure proper tensor shapes
            if values.dim() > 1:
                values = values.squeeze(-1)
            
            # Actor loss
            action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1))).squeeze()
            actor_loss = -(action_log_probs * advantages_tensor).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(values, returns_tensor)
            
            # Entropy loss (encourage exploration)
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            entropy_loss = -self.entropy_coeff * entropy
            
            # Total loss
            total_loss = actor_loss + critic_loss + entropy_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'total_loss': total_loss.item()
            }
            
        except Exception as e:
            print(f"Update error: {e}")
            return None
    
    def evaluate(self, env, num_episodes=100):
        """Evaluate the trained agent"""
        print(f"Starting single-thread A3C evaluation ({num_episodes} episodes)...")
        
        rewards = []
        lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.act(state, deterministic=True)
                state, reward, done, _ = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        results = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'success_rate': np.mean([r > 0 for r in rewards]),
            'all_rewards': rewards
        }
        
        print(f"Single-thread A3C Evaluation Results:")
        print(f"  Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Average Length: {results['avg_length']:.2f}")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        
        return results
    
    def save_model(self, filepath):
        """Save model"""
        torch.save(self.network.state_dict(), filepath)
        print(f"Single-thread A3C model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        self.network.load_state_dict(torch.load(filepath))
        print(f"Single-thread A3C model loaded from: {filepath}") 