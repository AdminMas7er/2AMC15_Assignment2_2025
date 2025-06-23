import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from agents.replaybuffer import ReplayBuffer
from pathlib import Path

class DQN_Network(nn.Module):
    def __init__(self, input_size,output_size):
        super(DQN_Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    def forward(self, state): 
        x = F.relu(self.fc1(state)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x) 
class DQNAgent:
    def __init__(self, state_size, action_size, target_update_freq, gamma, batch_size,
                 epsilon_start, epsilon_end, epsilon_decay, buffer_size,
                 learning_rate,device):
        self.device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.capacity = buffer_size
        self.lr = learning_rate
        self.q_network = DQN_Network(state_size, action_size).to(self.device)
        self.target_network = DQN_Network(state_size, action_size).to(self.device)
        self.replay_buffer = ReplayBuffer(self.capacity, self.batch_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.train_step = 0
    def _convert_obs_to_vector(self, obs_dict):
        """Converts a dictionary observation into a 9-element NumPy vector."""
        if obs_dict is None:
            # Return a zero vector of the correct size if observation is None
            return np.zeros(self.state_size, dtype=np.float32)

        # 1. Extract the values from the dictionary
        agent_pos = obs_dict["agent_pos"]
        agent_angle = obs_dict["agent_angle"]
        sensor_distances = obs_dict["sensor_distances"]
        has_order = obs_dict["has_order"]
        current_target_table = obs_dict["current_target_table"]

        # 2. Handle cases where values might be None or need type conversion
        if current_target_table is None:
            current_target_table = [0.0, 0.0] # Use a default value

        # 3. Create a single, flat NumPy array in a fixed order
        state_vector = np.concatenate([
            np.array(agent_pos, dtype=np.float32),
            np.array([agent_angle], dtype=np.float32),
            np.array(sensor_distances, dtype=np.float32),
            np.array([has_order], dtype=np.float32),
            np.array(current_target_table, dtype=np.float32)
        ])
        return state_vector
    def action(self,state,epsilon):
        if state is None:
            return random.randint(0, self.action_size - 1)

        state_vector = self._convert_obs_to_vector(state)

        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1).item())
    def observe(self,state,action,reward,next_state,done):
        state_vector = self._convert_obs_to_vector(state)
        next_state_vector = self._convert_obs_to_vector(next_state)

        self.replay_buffer.store(state_vector, action, reward, next_state_vector, float(done))
        self.optimize()
    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample()
        state_batch, action_batch, reward_batch, next_state_batch,done_batch = zip(*batch)
        state_tensor = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_tensor = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_tensor = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_tensor = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_tensor = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        current_q_values = self.q_network(state_tensor).gather(1, action_tensor)
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor).max(1)[0].unsqueeze(1)
            target_q_values=reward_tensor+self.gamma*next_q_values*(1-done_tensor)
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    def save(self, path: Path): # Ensure path is a Path object or string directory
        # Determine the full file path for the model
        if isinstance(path, str): # If path is a string, convert to Path
            path = Path(path)
        
        # If the provided path is a directory, append a default model name
        if path.is_dir():
            model_file_path = path / "dqn_model.pth"
        else: # Otherwise, assume path is the full desired file path
            model_file_path = path
            
        # Ensure the parent directory exists
        model_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.q_network.state_dict(), model_file_path) # Changed from self.model_net
        print(f"Model saved to {model_file_path}")

    def load(self, path): # path can be string or Path, to directory or file
        if isinstance(path, str):
            path = Path(path)
        
        if path.is_dir():
            model_file_path = path / "dqn_model.pth"
        else: # Assumes path is the full file path
            model_file_path = path
            
        if model_file_path.exists():
            self.q_network.load_state_dict(torch.load(model_file_path, map_location=self.device)) # Changed from self.model_net
            self.target_network.load_state_dict(self.q_network.state_dict()) # Changed from self.model_net to self.q_network
            self.q_network.eval() # Set q_network to eval mode, target_network is already in eval
            print(f"Model loaded from {model_file_path}")
        else:
            print(f"Warning: Model file not found at {model_file_path}")
