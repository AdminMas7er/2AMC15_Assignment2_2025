"""
SAC (Soft Actor-Critic) Agent Implementation
Supports discrete action spaces with automatic temperature tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class Actor(nn.Module):
    """Policy network (Actor)"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)

class Critic(nn.Module):
    """Q-network (Critic)"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    """SAC Agent for discrete action spaces"""
    
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, tau=0.005, 
                 alpha=0.2, buffer_size=100000, batch_size=256, hidden_size=256,
                 target_update_freq=1, auto_temp=True):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.auto_temp = auto_temp
        self.update_count = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Agent using device: {self.device}")
        
        # Networks
        self.actor = Actor(state_size, action_size, hidden_size).to(self.device)
        self.critic1 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.critic2 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.target_critic1 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.target_critic2 = Critic(state_size, action_size, hidden_size).to(self.device)
        
        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Automatic temperature tuning
        if self.auto_temp:
            self.target_entropy = -np.log(1.0 / action_size) * 0.98  # Heuristic value
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Experience replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.training_losses = {
            'actor_loss': [],
            'critic1_loss': [],
            'critic2_loss': [],
            'alpha_loss': [],
            'alpha_value': []
        }
    
    def act(self, state, deterministic=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state)
            
            if deterministic:
                # Choose action with highest probability
                action = torch.argmax(action_probs, dim=-1)
            else:
                # Sample from action distribution
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
            
            return action.cpu().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update_networks(self):
        """Update all networks"""
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch from replay buffer
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        states = torch.FloatTensor(batch.state).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)
        dones = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q-values
        current_q1 = self.critic1(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        current_q2 = self.critic2(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Next state action probabilities and Q-values
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_log_probs = torch.log(next_action_probs + 1e-8)
            
            next_q1 = self.target_critic1(next_states)
            next_q2 = self.target_critic2(next_states)
            next_q = torch.min(next_q1, next_q2)
            
            # Compute target Q-value with entropy term
            alpha = self.log_alpha.exp() if self.auto_temp else self.alpha
            next_v = (next_action_probs * (next_q - alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + self.gamma * next_v * (~dones)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor update
        action_probs = self.actor(states)
        log_probs = torch.log(action_probs + 1e-8)
        
        q1_values = self.critic1(states)
        q2_values = self.critic2(states)
        q_values = torch.min(q1_values, q2_values)
        
        alpha = self.log_alpha.exp() if self.auto_temp else self.alpha
        actor_loss = (action_probs * (alpha * log_probs - q_values)).sum(dim=-1).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Temperature parameter update
        alpha_loss = torch.tensor(0.0)
        if self.auto_temp:
            entropy = -(action_probs * log_probs).sum(dim=-1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update target networks
        if self.update_count % self.target_update_freq == 0:
            self.soft_update(self.target_critic1, self.critic1)
            self.soft_update(self.target_critic2, self.critic2)
        
        self.update_count += 1
        
        # Store training metrics
        losses = {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha_value': self.log_alpha.exp().item() if self.auto_temp else self.alpha
        }
        
        for key, value in losses.items():
            self.training_losses[key].append(value)
        
        return losses
    
    def soft_update(self, target, source):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train(self, env, num_episodes=1000, max_steps_per_episode=500, 
              update_freq=1, eval_freq=100, save_freq=100):
        """Train the SAC agent"""
        print(f"Starting SAC training for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Update networks
                if step % update_freq == 0:
                    self.update_networks()
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Print progress
            if episode % eval_freq == 0:
                recent_rewards = episode_rewards[-eval_freq:] if len(episode_rewards) >= eval_freq else episode_rewards
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(episode_lengths[-eval_freq:]) if len(episode_lengths) >= eval_freq else np.mean(episode_lengths)
                
                current_alpha = self.log_alpha.exp().item() if self.auto_temp else self.alpha
                buffer_size = len(self.memory)
                
                print(f"Episode {episode}/{num_episodes}")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Average Length: {avg_length:.2f}")
                print(f"  Alpha (Temperature): {current_alpha:.4f}")
                print(f"  Buffer Size: {buffer_size}")
                print("-" * 50)
            
            # Save model periodically
            if episode % save_freq == 0 and episode > 0:
                self.save_model(f'models/sac_model_ep{episode}.pth')
        
        print("SAC training completed!")
        return episode_rewards, episode_lengths
    
    def evaluate(self, env, num_episodes=100):
        """Evaluate the trained agent"""
        print("Starting SAC evaluation...")
        
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
        
        print(f"SAC Evaluation Results ({num_episodes} episodes):")
        print(f"  Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Average Length: {results['avg_length']:.2f}")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        
        return results
    
    def save_model(self, filepath):
        """Save the trained models"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha if self.auto_temp else None,
            'training_losses': self.training_losses
        }, filepath)
        print(f"SAC model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained models"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        if self.auto_temp and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
        
        if 'training_losses' in checkpoint:
            self.training_losses = checkpoint['training_losses']
        
        print(f"SAC model loaded from: {filepath}") 