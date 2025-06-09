
import numpy as np
import random

from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module): #nn.Module is like a base class/template for neural networks
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, action_bins=7):
        # We will note the hyperparameters and settings below
        self.max_speed = 0.2
        self.max_turn = np.radians(30)
        self.speed_levels = np.linspace(0, self.max_speed, action_bins)
        self.turn_levels = np.linspace(-self.max_turn, self.max_turn, action_bins)
        self.actions = [(s, r) for s in self.speed_levels for r in self.turn_levels] #Discretizing actions for the DQN


        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.memory_size = 10000
        self.target_update_frequency = 4 # I see very different values used online varying from 4 to 1000 so this should be checked
        self.learning_rate = 0.001

        #Initialize Q-networks
        self.input_dim = 3 + 2 + 1  # sensor distances (3), agent_pos (2), agent_angle (1)
        self.output_dim = len(self.actions) #the amount of possible actions using the set discretization action bins
        self.model_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim) #This helps DQN for stability
        self.target_net.load_state_dict(self.model_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate) #includes learning rate
        self.memory = deque(maxlen=self.memory_size)

    def _obs_to_tensor(self, obs):
        """"Converts an observation into a tensor.??????"""
        vec = np.concatenate([
            obs["sensor_distances"],
            obs["agent_pos"],
            [obs["agent_angle"]]
        ])
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

    def select_action(self, observation):
    """Function to choose an action using the epsilon-greedy policy"""
        if np.random.rand() < self.epsilon: #Exploration
            return random.choice(self.actions)
        else: #Exploitation
            with torch.no_grad():
                state = self._obs_to_tensor(observation)
                q_values = self.model_net(state)
                action_index = torch.argmax(q_values).item()
                return self.actions[action_index]



    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)
