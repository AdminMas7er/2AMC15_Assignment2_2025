import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

class DQN_Network(nn.Module):
    def __init__(self, input_size,output_size):
        super(DQN_Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_size)
    def forward(self, state): 
        x = F.relu(self.fc1(state)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x) 
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
class DQNAgent:
    def __init__(self, state_size,action_size,target_update_freq=1000,gamma=0.9,batch_size=32,epsilon_start=1.0,epsilon_end=0.01,epsilon_decay=0.995,buffer_size=10000,learning_rate=0.00025,episodes=1000):
        self.state_size = state_size
        self.episode_length=episodes
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon=epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.capacity= buffer_size
        self.lr=learning_rate
        self.q_network = DQN_Network(state_size, action_size)
        self.target_network = DQN_Network(state_size, action_size)
        self.replay_buffer = ReplayBuffer(self.capacity)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def action(self,state,epsilon): # Make sure state is a preprocessed numpy array or tensor
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad(): # Use no_grad for inference
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        # Ensure states pushed to buffer are preprocessed numpy arrays
        state_batch,action_batch,reward_batch,next_state_batch,done_batch=zip(*transitions)
        
        state_batch = torch.FloatTensor(np.array(state_batch)) # Convert list of arrays to single tensor
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1) # Ensure correct shape for loss
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)) # Convert list of arrays to single tensor
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1) # Ensure correct shape

        q_values = self.q_network(state_batch).gather(1, action_batch)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0].unsqueeze(1) # Ensure correct shape
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, target_q_values.detach()) # detach target_q_values
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

    def train(self, env):
        pass




