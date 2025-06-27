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
    """Enhanced Policy network (Actor) with stable architecture"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Additional layer
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)
        return F.softmax(logits, dim=-1)

class Critic(nn.Module):
    """Enhanced Q-network (Critic) with stable architecture"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Additional layer
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

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
        
        # Automatic temperature tuning with controlled initialization
        if self.auto_temp:
            self.target_entropy = -np.log(1.0 / action_size) * 0.8  # Balanced target
            # Initialize with moderate temperature
            self.log_alpha = torch.tensor([np.log(0.1)], requires_grad=True, device=self.device, dtype=torch.float32)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr * 0.1)  # Much slower alpha learning
        
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
        self.collisions = []
    
    def state_to_vector(self, obs):
        """
        Advanced state representation with target-directed navigation
        """
        if isinstance(obs, dict):
            # Agent position (normalized to [0,1])
            agent_pos = np.array(obs["agent_pos"]) / 10.0  # [x, y] normalized
            
            # Agent heading (cosθ, sinθ) 
            heading = np.array(obs["heading"])  # [cos, sin]
            
            # Current target as one-hot encoding (assume max 5 tables)
            target_id = int(obs["current_target"])
            target_onehot = np.zeros(5)  # Max 5 tables
            if 0 <= target_id < 5:
                target_onehot[target_id] = 1.0
            
            # Enhanced features for better navigation
            pos_real = np.array(obs["agent_pos"])
            
            # Distance to boundaries (normalized)
            dist_to_boundaries = np.array([
                pos_real[0] / 10.0,          # distance to left boundary
                (10.0 - pos_real[0]) / 10.0, # distance to right boundary  
                pos_real[1] / 10.0,          # distance to bottom boundary
                (10.0 - pos_real[1]) / 10.0  # distance to top boundary
            ])
            
            # Target direction estimation (crucial for navigation!)
            # Use fixed table positions based on standard restaurant layout
            estimated_table_positions = [
                [2.5, 2.5],  # Table 0: bottom-left
                [7.5, 2.5],  # Table 1: bottom-right
                [2.5, 7.5],  # Table 2: top-left
                [7.5, 7.5],  # Table 3: top-right  
                [5.0, 5.0],  # Table 4: center
            ]
            
            if target_id < len(estimated_table_positions):
                target_pos = np.array(estimated_table_positions[target_id])
                # Direction to target
                to_target_vec = target_pos - pos_real
                target_distance = np.linalg.norm(to_target_vec) / 7.07  # normalized
                to_target_vec = to_target_vec / (np.linalg.norm(to_target_vec) + 1e-8)
                
                # Heading alignment with target direction
                heading_to_target = np.dot(heading, to_target_vec)
                
                # Angular difference to target
                target_angle = np.arctan2(to_target_vec[1], to_target_vec[0])
                agent_angle = np.arctan2(heading[1], heading[0])
                angular_diff = np.sin(target_angle - agent_angle)  # sin gives signed shortest angle
            else:
                target_distance = 0.5
                heading_to_target = 0.0
                angular_diff = 0.0
            
            # Wall proximity warnings
            wall_proximity = np.array([
                1.0 if pos_real[0] < 1.0 else 0.0,  # near left wall
                1.0 if pos_real[0] > 9.0 else 0.0,  # near right wall
                1.0 if pos_real[1] < 1.0 else 0.0,  # near bottom wall
                1.0 if pos_real[1] > 9.0 else 0.0,  # near top wall
            ])
            
            return np.concatenate([
                agent_pos,                    # [x, y] - 2D
                heading,                      # [cos, sin] - 2D  
                target_onehot,                # [t0, t1, t2, t3, t4] - 5D
                dist_to_boundaries,           # [left, right, bottom, top] - 4D
                np.array([target_distance]),  # [target_dist] - 1D
                np.array([heading_to_target]), # [heading_alignment] - 1D
                np.array([angular_diff]),      # [angular_difference] - 1D
                wall_proximity                # [wall_warnings] - 4D
            ]).astype(np.float32)  # Total: 20D
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
            
            # Return discrete action index for enviroment_final compatibility
            return action_idx.cpu().item()
    
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
        """Store experience in replay buffer with reward shaping"""
        state_vec = self.state_to_vector(state)
        next_state_vec = self.state_to_vector(next_state)
        
        # Convert continuous action back to discrete index for storage
        if isinstance(action, tuple):
            action_idx = self._continuous_to_discrete_action(action)
        else:
            action_idx = action
        
        # Internal reward shaping for better learning
        shaped_reward = self._shape_reward(state, next_state, reward, done)
            
        self.memory.push(state_vec, action_idx, shaped_reward, next_state_vec, done)
    
    def _shape_reward(self, state, next_state, original_reward, done):
        """
        Advanced reward shaping with target-directed navigation rewards
        """
        shaped_reward = float(original_reward)
        
        if isinstance(state, dict) and isinstance(next_state, dict):
            curr_pos = np.array(state["agent_pos"], dtype=np.float32)
            next_pos = np.array(next_state["agent_pos"], dtype=np.float32)
            target_id = int(state["current_target"])
            
            # Target-directed progress reward
            estimated_table_positions = [
                [2.5, 2.5],  # Table 0: bottom-left
                [7.5, 2.5],  # Table 1: bottom-right
                [2.5, 7.5],  # Table 2: top-left
                [7.5, 7.5],  # Table 3: top-right  
                [5.0, 5.0],  # Table 4: center
            ]
            
            if target_id < len(estimated_table_positions):
                target_pos = np.array(estimated_table_positions[target_id], dtype=np.float32)
                
                # Progress towards target
                curr_dist_to_target = float(np.linalg.norm(curr_pos - target_pos))
                next_dist_to_target = float(np.linalg.norm(next_pos - target_pos))
                target_progress = curr_dist_to_target - next_dist_to_target
                
                # Strong reward for getting closer to target
                shaped_reward += target_progress * 2.0
                
                # Bonus for being very close to target
                if next_dist_to_target < 1.0:
                    shaped_reward += (1.0 - next_dist_to_target) * 0.5
            
            # Movement efficiency
            movement = float(np.linalg.norm(next_pos - curr_pos))
            
            # Anti-stagnation penalty
            if movement < 0.05:  # Very small movement
                shaped_reward -= 0.1
            
            # Directional movement bonus
            if movement > 0.3:  # Good movement
                shaped_reward += 0.05
            
            # Wall avoidance reward
            for pos in [next_pos]:
                if pos[0] < 0.5 or pos[0] > 9.5 or pos[1] < 0.5 or pos[1] > 9.5:
                    shaped_reward -= 0.2  # Penalty for being near walls
            
            # Success amplification (critical for learning)
            if done and original_reward > 40:  # Success (original +50)
                shaped_reward += 15.0  # Strong success bonus
            
            # Collision handling - less severe but still discouraged
            if original_reward < -2.5:  # Likely collision
                shaped_reward = max(shaped_reward, -1.5)  # Cap collision penalty
        
        return np.float32(shaped_reward)
    
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
        
        states = torch.FloatTensor(np.array(batch.state, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward, dtype=np.float32)).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state, dtype=np.float32)).to(self.device)
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
        
        # Temperature parameter update with constraints
        alpha_loss = torch.tensor(0.0)
        if self.auto_temp:
            entropy = -(action_probs * log_probs).sum(dim=-1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Constrain alpha to reasonable range
            with torch.no_grad():
                self.log_alpha.clamp_(-5.0, 2.0)  # alpha in [exp(-5), exp(2)] = [0.007, 7.4]
        
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
            'episode_lengths': self.episode_lengths,
            'collisions': self.collisions
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
        if 'collissions' in checkpoint:
            self.episode_lengths = checkpoint['collisions']
        
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