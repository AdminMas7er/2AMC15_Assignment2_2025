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
        self.q_network = DQN_Network(state_size, action_size)
        self.q_network.to(self.device)  
        self.target_network = DQN_Network(state_size, action_size)
        self.target_network.to(self.device)
        self.replay_buffer = ReplayBuffer(self.capacity, self.batch_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.train_step = 0
    def action(self,state,epsilon): 
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Ensure state is on the correct device
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1 ).item())
    def observe(self,state,action,reward,next_state,done):
        self.replay_buffer.store(state, action, reward, next_state, float(done))
        self.optimize()
    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample()
        # valid_batch = []
        # for trans in batch:
        #     s, a, r, ns = trans
        #     if (
        #             s is not None and
        #             ns is not None and
        #             isinstance(s, np.ndarray) and
        #             isinstance(ns, np.ndarray) and
        #             s.size > 0 and
        #             ns.size > 0 and
        #             not np.any(np.isnan(s)) and
        #             not np.any(np.isnan(ns))
        #     ):
        #         valid_batch.append(trans)
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
# class Agent(DQNAgent):
#         def __init__(self, state_size, action_size):
#             super().__init__(state_size, action_size)

#         def state_to_vector(self, obs):
#             agent_pos = obs["agent_pos"]  # (2,)
#             # agent_angle = np.array([obs["agent_angle"]])  # (1,)
#             # sensor_distances = np.array(obs["sensor_distances"])  # (3,)
#             # pickup_point = obs["pickup_point"]  # (2,)
#             has_order = np.array([float(obs["has_order"])])  # (1,)

#             #One hot encoding of target table
#             num_tables = len(obs["target_tables"])
#             target_idx = None
#             for idx, table in enumerate(obs["target_tables"]):
#                 if np.allclose(table, obs["current_target_table"], atol=1e-2):
#                     target_idx = idx
#                     break

#             target_onehot = np.zeros(num_tables)
#             if target_idx is not None:
#                 target_onehot[target_idx] = 1.0

#             return np.concatenate([
#                 agent_pos,
#                 has_order,
#                 target_onehot
#             ])

      
#         def take_action(self, observation): # This is called by train_dqn_final.py
#              state_vector = self.state_to_vector(observation)
#              return self.select_action(state_vector)

   
#         def observe_transition(self, state, action, reward, next_state, done):
#             state_vector = self.state_to_vector(state)
#             next_state_vector = self.state_to_vector(next_state)

#             valid = all([
#                 state_vector is not None,
#                 next_state_vector is not None,
#                 not np.any(np.isnan(state_vector)),
#                 not np.any(np.isnan(next_state_vector)),
#                 state_vector.size > 0,
#                 next_state_vector.size > 0
#             ])

#             if valid:
#                 self.remember(state_vector, action, reward, next_state_vector)
#                 self.train()