import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from replaybuffer import ReplayBuffer
from delivery_environment import DeliveryEnvironment

class DQN(nn.Module):
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
    def __init__(self, state_size, action_size):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.memory_size = 10000
        self.target_update_frequency = 4
        self.learning_rate = 0.001

        self.model_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.model_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.model_net.parameters(), self.learning_rate)
        self.memory = ReplayBuffer(self.memory_size, self.batch_size)
        self.train_step = 0
        self.action_size = action_size

    def _state_to_tensor(self, state):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                q_values = self.model_net(state_tensor)
                return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state):
        self.memory.store(state, action, reward, next_state)

    def train(self):
        if self.memory.currentSize < self.batch_size:
            return

        batch = self.memory.sample()
        state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch)

        current_q_values = self.model_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.model_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        torch.save(self.model_net.state_dict(), filepath)

    def load(self, filepath):
        self.model_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.model_net.state_dict())

# Wrapper class to match your train.py interface
class Agent(DQNAgent):
    def __init__(self):
        # Use DeliveryEnvironment to get state and action sizes
        dummy_env = DeliveryEnvironment(enable_gui=False)
        state_size = dummy_env.get_state_size()
        action_size = dummy_env.get_action_size()
        super().__init__(state_size, action_size)

        # Initialize state
        self.last_state = None
        self.last_action = None

    def take_action(self, observation):
        """Wrapper function for train.py"""
        # If this is the first step, no training yet
        if self.last_state is not None and self.last_action is not None:
            # Dummy reward (you may want to pass true reward here if available)
            reward = 0
            # Dummy next state → observation is already next_state in your train.py loop
            self.remember(self.last_state, self.last_action, reward, observation)
            self.train()

        action = self.select_action(observation)

        # Save current state/action to store transition at next call
        self.last_state = observation
        self.last_action = action

        return action
