import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import threading
import time
from collections import deque

class ActorCriticNetwork(nn.Module):
    """Actor-Critic Network"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor layer (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic layer (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared_layers(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

class Worker(threading.Thread):
    """A3C Worker Thread"""
    def __init__(self, worker_id, global_network, optimizer, env_fn, 
                 max_episodes=1000, gamma=0.99, entropy_coeff=0.01):
        super(Worker, self).__init__()
        
        self.worker_id = worker_id
        self.global_network = global_network
        self.optimizer = optimizer
        self.env = env_fn()
        
        # Local network
        self.local_network = ActorCriticNetwork(
            self.env.get_state_size(), 
            self.env.get_action_size()
        )
        
        # Hyperparameters
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.update_frequency = 20  # Update every 20 steps
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
    def sync_with_global(self):
        """Sync local network with global network"""
        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[i + 1]
            
            delta = rewards[i] + gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def run(self):
        """Worker thread main loop"""
        episode = 0
        
        while episode < self.max_episodes:
            self.sync_with_global()
            
            # Collect experience
            states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
            
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.update_frequency):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                with torch.no_grad():
                    action_probs, value = self.local_network(state_tensor)
                
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                next_state, reward, done, _ = self.env.step(action.item())
                
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
                    _, next_value = self.local_network(torch.FloatTensor(state).unsqueeze(0))
                    next_value = next_value.item()
            
            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values, next_value, dones, self.gamma)
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)
            advantages_tensor = torch.FloatTensor(advantages)
            returns_tensor = torch.FloatTensor(returns)
            
            # Compute loss and update global network
            self.update_global_network(states_tensor, actions_tensor, 
                                     advantages_tensor, returns_tensor)
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode += 1
                
                if episode % 100 == 0:
                    avg_reward = np.mean(self.episode_rewards)
                    avg_length = np.mean(self.episode_lengths)
                    print(f"Worker {self.worker_id}, Episode {episode}, "
                          f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
    
    def update_global_network(self, states, actions, advantages, returns):
        """Update global network"""
        self.optimizer.zero_grad()
        
        # Forward pass
        action_probs, values = self.local_network(states)
        
        # Actor loss
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        actor_loss = -(action_log_probs * advantages).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy loss (encourage exploration)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        entropy_loss = -self.entropy_coeff * entropy
        
        # Total loss
        total_loss = actor_loss + critic_loss + entropy_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), max_norm=0.5)
        
        # Update global network
        for local_param, global_param in zip(self.local_network.parameters(), 
                                           self.global_network.parameters()):
            if global_param.grad is None:
                global_param._grad = local_param.grad
            else:
                global_param._grad += local_param.grad
        
        self.optimizer.step()

class A3CAgent:
    """A3C Agent"""
    def __init__(self, state_size, action_size, num_workers=4, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.num_workers = num_workers
        
        # Global network
        self.global_network = ActorCriticNetwork(state_size, action_size)
        self.global_network.share_memory()  # Share memory
        
        # Optimizer
        self.optimizer = optim.Adam(self.global_network.parameters(), lr=lr)
        
        # Worker threads
        self.workers = []
        
    def train(self, env_fn, max_episodes=1000):
        """Train A3C agent"""
        # Create worker threads
        for i in range(self.num_workers):
            worker = Worker(
                worker_id=i,
                global_network=self.global_network,
                optimizer=self.optimizer,
                env_fn=env_fn,
                max_episodes=max_episodes // self.num_workers
            )
            self.workers.append(worker)
        
        # Start all worker threads
        for worker in self.workers:
            worker.start()
        
        # Wait for all threads to complete
        for worker in self.workers:
            worker.join()
        
        print("A3C training completed!")
    
    def save_model(self, filepath):
        """Save model"""
        torch.save(self.global_network.state_dict(), filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        self.global_network.load_state_dict(torch.load(filepath))
        print(f"Model loaded from: {filepath}")
    
    def act(self, state, deterministic=False):
        """Select action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, _ = self.global_network(state_tensor)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=1).item()
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        
        return action 