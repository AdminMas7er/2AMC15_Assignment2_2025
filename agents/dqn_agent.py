import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

# Experience structure for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """DQN Neural Network"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state):
        return self.network(state)

class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays first to avoid slow tensor creation warning
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.BoolTensor(np.array([e.done for e in experiences]))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent"""
    def __init__(self, state_size, action_size, lr=0.00025, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 batch_size=32, target_update_freq=1000, buffer_size=10000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # ε-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training counter
        self.update_count = 0
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.losses = deque(maxlen=1000)
        
        # Sync target network
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state, deterministic=False):
        """ε-greedy action selection"""
        if deterministic or random.random() > self.epsilon:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        else:
            action = random.randrange(self.action_size)
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        """Store experience and learn"""
        # Store experience
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # If we have enough experience, start learning
        if len(self.replay_buffer) > self.batch_size:
            self.learn()
    
    def learn(self):
        """Learn from experience replay"""
        # Sample experience
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Calculate current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calculate target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        # Calculate loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Record loss
        self.losses.append(loss.item())
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay ε
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, num_episodes=2000, max_steps_per_episode=500):
        """Train DQN agent"""
        print("Starting DQN training...")
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                self.step(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-100:])
                avg_length = np.mean(list(self.episode_lengths)[-100:])
                avg_loss = np.mean(list(self.losses)[-100:]) if self.losses else 0
                
                print(f"Episode {episode}/{num_episodes}")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Average Length: {avg_length:.2f}")
                print(f"  Average Loss: {avg_loss:.4f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
                print(f"  Replay Buffer Size: {len(self.replay_buffer)}")
                print("-" * 50)
        
        print("DQN training completed!")
    
    def evaluate(self, env, num_episodes=100, render=False):
        """Evaluate agent performance"""
        print("Starting evaluation...")
        
        eval_rewards = []
        eval_lengths = []
        
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
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        avg_length = np.mean(eval_lengths)
        
        print(f"Evaluation Results ({num_episodes} episodes):")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Average Length: {avg_length:.2f}")
        print(f"  Success Rate: {np.mean([r > 0 for r in eval_rewards]):.2%}")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_length': avg_length,
            'success_rate': np.mean([r > 0 for r in eval_rewards]),
            'all_rewards': eval_rewards
        }
    
    def save_model(self, filepath):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, filepath)
        print(f"DQN model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
        
        print(f"DQN model loaded from: {filepath}") 