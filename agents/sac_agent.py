"""
SAC (Soft Actor-Critic) Agent Implementation for Restaurant Delivery
Qi's implementation for 2AMC15 Assignment 2 - designed to outperform Stefan's DQN baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os
from pathlib import Path

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class Actor(nn.Module):
    """Policy network (Actor) with discrete action support"""
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
    """Experience replay buffer for SAC"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    """
    SAC Agent for discrete action spaces - Qi's implementation
    Designed to outperform DQN baseline with better sample efficiency and stability
    """
    
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, tau=0.005, 
                 alpha=0.2, buffer_size=100000, batch_size=256, hidden_size=256,
                 target_update_freq=1, auto_temp=True, device=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.auto_temp = auto_temp
        self.update_count = 0
        
        # Device configuration
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Agent using device: {self.device}")
        
        # Networks
        self.actor = Actor(state_size, action_size, hidden_size).to(self.device)
        self.critic1 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.critic2 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.target_critic1 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.target_critic2 = Critic(state_size, action_size, hidden_size).to(self.device)
        
        # Initialize target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Automatic temperature tuning
        if self.auto_temp:
            self.target_entropy = -np.log(1.0 / action_size) * 0.98  # Target entropy heuristic
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
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
    
    def state_to_vector(self, obs):
        """
        Convert environment observation to state vector
        Optimized state representation for SAC
        """
        if isinstance(obs, dict):
            # Agent position and orientation
            agent_pos = np.array(obs["agent_pos"])  # [x, y]
            agent_angle = np.array([obs["agent_angle"]])  # [angle]
            
            # Sensor distances (normalized)
            sensor_distances = np.array(obs["sensor_distances"]) / 5.0  # Normalize by max range
            
            # Task state - handle missing 'has_order' key
            has_order = np.array([float(obs.get("has_order", 0.0))])
            
            # Distance to pickup point (normalized)
            if "pickup_point" in obs:
                pickup_dist = np.linalg.norm(agent_pos - obs["pickup_point"]) / 10.0
            else:
                pickup_dist = 1.0  # Default max distance
            
            # Distance to current target (if exists)
            if "current_target_table" in obs and obs["current_target_table"] is not None:
                target_dist = np.linalg.norm(agent_pos - obs["current_target_table"]) / 10.0
            else:
                target_dist = 1.0  # Max distance when no target
            
            # Closest table distance for obstacle avoidance
            if "target_tables" in obs and len(obs["target_tables"]) > 0:
                table_distances = [np.linalg.norm(agent_pos - table) for table in obs["target_tables"]]
                closest_table_dist = min(table_distances) / 10.0
            else:
                closest_table_dist = 1.0
            
            return np.concatenate([
                agent_pos,          # [x, y] - 2D
                agent_angle,        # [angle] - 1D
                sensor_distances,   # [front, left, right] - 3D
                has_order,          # [has_order] - 1D
                np.array([pickup_dist]),      # [pickup_distance] - 1D
                np.array([target_dist]),      # [target_distance] - 1D
                np.array([closest_table_dist]) # [obstacle_distance] - 1D
            ]).astype(np.float32)  # Total: 10D
        else:
            return obs.astype(np.float32)
    
    def act(self, observation, deterministic=False):
        """Select action using current policy"""
        state = self.state_to_vector(observation)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state)
            
            if deterministic:
                # Choose action with highest probability
                action_idx = torch.argmax(action_probs, dim=-1)
            else:
                # Sample from action distribution
                action_dist = torch.distributions.Categorical(action_probs)
                action_idx = action_dist.sample()
            
            # Convert discrete action to continuous action
            return self._discrete_to_continuous_action(action_idx.cpu().item())
    
    def _discrete_to_continuous_action(self, action_idx):
        """Convert discrete action index to continuous (velocity, rotation) action"""
        # Define discrete action mappings
        action_map = {
            0: (0.0, 0.0),      # Do nothing
            1: (0.0, -0.5),     # Rotate left
            2: (0.0, 0.5),      # Rotate right
            3: (0.5, 0.0),      # Move forward
            4: (-0.5, 0.0),     # Move backward
        }
        return action_map.get(action_idx, (0.0, 0.0))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        state_vec = self.state_to_vector(state)
        next_state_vec = self.state_to_vector(next_state)
        
        # Convert continuous action back to discrete index for storage
        if isinstance(action, tuple):
            action_idx = self._continuous_to_discrete_action(action)
        else:
            action_idx = action
            
        self.memory.push(state_vec, action_idx, reward, next_state_vec, done)
    
    def _continuous_to_discrete_action(self, action):
        """Convert continuous (velocity, rotation) action to discrete index"""
        velocity, rotation = action
        
        # Find closest matching discrete action
        action_map = {
            0: (0.0, 0.0),      # Do nothing
            1: (0.0, -0.5),     # Rotate left
            2: (0.0, 0.5),      # Rotate right
            3: (0.5, 0.0),      # Move forward
            4: (-0.5, 0.0),     # Move backward
        }
        
        best_idx = 0
        best_distance = float('inf')
        
        for idx, (v, r) in action_map.items():
            distance = abs(velocity - v) + abs(rotation - r)
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
                
        return best_idx
    
    def update_networks(self):
        """Update all networks (main SAC update step)"""
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch from replay buffer
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
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
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
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
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
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
    
    def save_model(self, filepath):
        """Save the trained models"""
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha if self.auto_temp else None,
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, filepath)
        print(f"SAC model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained models"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        if self.auto_temp and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
        
        if 'training_losses' in checkpoint:
            self.training_losses = checkpoint['training_losses']
        if 'episode_rewards' in checkpoint:
            self.episode_rewards = checkpoint['episode_rewards']
        if 'episode_lengths' in checkpoint:
            self.episode_lengths = checkpoint['episode_lengths']
        
        print(f"SAC model loaded from: {filepath}")

class Agent:
    """Wrapper class for easy integration with training scripts"""
    def __init__(self):
        # Will be initialized with proper parameters in training script
        self.sac_agent = None
    
    def take_action(self, observation):
        if self.sac_agent:
            return self.sac_agent.act(observation)
        else:
            return random.randint(0, 4)  # Random action if not initialized 