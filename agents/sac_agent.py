"""
SAC (Soft Actor-Critic) Agent Implementation for Restaurant Delivery
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
    """Policy network (Actor)"""
    def __init__(self, state_size, action_size, hidden_size=512):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc5 = nn.Linear(hidden_size//4, action_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        logits = self.fc5(x)
        return F.softmax(logits, dim=-1)

class Critic(nn.Module):
    """Q-network (Critic)"""
    def __init__(self, state_size, action_size, hidden_size=512):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc5 = nn.Linear(hidden_size//4, action_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)

class ReplayBuffer:
    """Experience replay buffer with success prioritization"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.success_buffer = deque(maxlen=capacity//4)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        if done and reward > 40:
            self.success_buffer.append(experience)
    
    def sample(self, batch_size):
        if len(self.success_buffer) > 0 and len(self.buffer) > batch_size:
            success_ratio = min(0.3, len(self.success_buffer) / len(self.buffer))
            num_success = int(batch_size * success_ratio)
            num_regular = batch_size - num_success
            
            if num_success > 0 and len(self.success_buffer) >= num_success:
                success_samples = random.sample(list(self.success_buffer), num_success)
            else:
                success_samples = []
                num_regular = batch_size
                
            regular_samples = random.sample(list(self.buffer), num_regular)
            return success_samples + regular_samples
        else:
            return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    """SAC Agent for discrete action spaces"""
    
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
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = Actor(state_size, action_size, hidden_size).to(self.device)
        self.critic1 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.critic2 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.target_critic1 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.target_critic2 = Critic(state_size, action_size, hidden_size).to(self.device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.999)
        self.critic1_scheduler = optim.lr_scheduler.ExponentialLR(self.critic1_optimizer, gamma=0.999)
        self.critic2_scheduler = optim.lr_scheduler.ExponentialLR(self.critic2_optimizer, gamma=0.999)
        
        # Temperature tuning
        if self.auto_temp:
            self.target_entropy = -np.log(1.0 / action_size) * 0.6
            self.log_alpha = torch.tensor([np.log(0.05)], requires_grad=True, device=self.device, dtype=torch.float32)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr * 0.02)
            self.alpha_min = 0.005
        
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.training_losses = {
            'actor_loss': [],
            'critic1_loss': [],
            'critic2_loss': [],
            'alpha_loss': [],
            'alpha_value': []
        }
        
        self.episode_rewards = []
        self.episode_lengths = []
    
    def state_to_vector(self, obs):
        """Convert observation to state vector"""
        if isinstance(obs, dict):
            agent_pos = np.array(obs["agent_pos"]) / 10.0
            heading = np.array(obs["heading"])
            
            target_id = int(obs["current_target"])
            target_onehot = np.zeros(5)
            if 0 <= target_id < 5:
                target_onehot[target_id] = 1.0
            
            pos_real = np.array(obs["agent_pos"])
            
            dist_to_boundaries = np.array([
                pos_real[0] / 10.0,
                (10.0 - pos_real[0]) / 10.0,
                pos_real[1] / 10.0,
                (10.0 - pos_real[1]) / 10.0
            ])
            
            table_positions = np.array([
                [2.0, 2.0], [8.0, 2.0], [5.0, 5.0], [2.0, 8.0], [8.0, 8.0]
            ])
            
            if 0 <= target_id < 5:
                target_pos = table_positions[target_id]
                to_target = target_pos - pos_real
                target_distance = np.linalg.norm(to_target) / 10.0
                to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-8)
                target_alignment = np.dot(heading, to_target_norm)
            else:
                target_distance = 0.5
                target_alignment = 0.0
                to_target_norm = np.array([0.0, 0.0])
            
            center_pos = np.array([5.0, 5.0])
            dist_to_center = np.linalg.norm(pos_real - center_pos) / 7.07
            
            wall_proximity = np.array([
                1.0 if pos_real[0] < 1.0 else 0.0,
                1.0 if pos_real[0] > 9.0 else 0.0,
                1.0 if pos_real[1] < 1.0 else 0.0,
                1.0 if pos_real[1] > 9.0 else 0.0,
            ])
            
            pos_encoding = np.array([
                np.sin(pos_real[0] * np.pi / 10.0),
                np.cos(pos_real[0] * np.pi / 10.0),
                np.sin(pos_real[1] * np.pi / 10.0),
                np.cos(pos_real[1] * np.pi / 10.0),
            ])
            
            return np.concatenate([
                agent_pos,
                heading,
                target_onehot,
                dist_to_boundaries,
                np.array([target_distance]),
                np.array([target_alignment]),
                to_target_norm,
                np.array([dist_to_center]),
                wall_proximity,
                pos_encoding,
            ]).astype(np.float32)
        else:
            return obs.astype(np.float32)
    
    def act(self, observation, deterministic=False):
        """Select action using current policy"""
        state = self.state_to_vector(observation)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state)
            
            if deterministic:
                action_idx = torch.argmax(action_probs, dim=-1)
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action_idx = action_dist.sample()
            
            return action_idx.cpu().item()
    
    def _discrete_to_continuous_action(self, action_idx):
        """Convert discrete action index to continuous action"""
        action_map = {
            0: (0.0, 0.0),
            1: (0.0, -0.5),
            2: (0.0, 0.5),
            3: (0.5, 0.0),
            4: (-0.5, 0.0),
        }
        return action_map.get(action_idx, (0.0, 0.0))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        state_vec = self.state_to_vector(state)
        next_state_vec = self.state_to_vector(next_state)
        
        if isinstance(action, tuple):
            action_idx = self._continuous_to_discrete_action(action)
        else:
            action_idx = action
        
        shaped_reward = self._shape_reward(state, next_state, reward, done)
        self.memory.push(state_vec, action_idx, shaped_reward, next_state_vec, done)
    
    def _shape_reward(self, state, next_state, original_reward, done):
        """Apply reward shaping"""
        shaped_reward = float(original_reward)
        
        if isinstance(state, dict) and isinstance(next_state, dict):
            curr_pos = np.array(state["agent_pos"], dtype=np.float32)
            next_pos = np.array(next_state["agent_pos"], dtype=np.float32)
            
            center = np.array([5.0, 5.0], dtype=np.float32)
            curr_dist_to_center = np.linalg.norm(curr_pos - center)
            next_dist_to_center = np.linalg.norm(next_pos - center)
            
            progress = float(curr_dist_to_center - next_dist_to_center)
            shaped_reward += progress * 0.5
            
            movement = float(np.linalg.norm(next_pos - curr_pos))
            if movement < 0.05:
                shaped_reward -= 0.05
            
            if movement > 0.5:
                shaped_reward += 0.1
            
            if done and original_reward > 40:
                shaped_reward += 10.0
            
            if original_reward < -2.5:
                shaped_reward = max(shaped_reward, -2.0)
        
        return np.float32(shaped_reward)
    
    def _continuous_to_discrete_action(self, action):
        """Convert continuous action to discrete index"""
        velocity, rotation = action
        
        action_map = {
            0: (0.0, 0.0),
            1: (0.0, -0.5),
            2: (0.0, 0.5),
            3: (0.5, 0.0),
            4: (-0.5, 0.0),
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
        """Update all networks"""
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch
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
        
        # Next state values
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_log_probs = torch.log(next_action_probs + 1e-8)
            
            next_q1 = self.target_critic1(next_states)
            next_q2 = self.target_critic2(next_states)
            next_q = torch.min(next_q1, next_q2)
            
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
            
            with torch.no_grad():
                alpha_min_log = np.log(self.alpha_min)
                self.log_alpha.clamp_(alpha_min_log, 1.0)
        
        # Soft update target networks
        if self.update_count % self.target_update_freq == 0:
            self.soft_update(self.target_critic1, self.critic1)
            self.soft_update(self.target_critic2, self.critic2)
        
        # Update learning rates
        if self.update_count % 100 == 0:
            self.actor_scheduler.step()
            self.critic1_scheduler.step()
            self.critic2_scheduler.step()
        
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

class Agent:
    """Wrapper class for easy integration with training scripts"""
    def __init__(self):
        self.sac_agent = None
    
    def take_action(self, observation):
        if self.sac_agent:
            return self.sac_agent.act(observation)
        else:
            return random.randint(0, 4) 