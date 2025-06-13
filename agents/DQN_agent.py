import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from .replaybuffer import ReplayBuffer  # relative import

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
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.memory_size = 10000
        self.target_update_frequency = 4
        self.learning_rate = 1e-4

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
        """Select an action by epsilon greedy policy"""
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
        if len(self.memory.buffer) < self.batch_size:
            return

        batch = self.memory.sample()

        valid_batch = []
        for trans in batch:
            s, a, r, ns = trans
            if (
                s is not None and
                ns is not None and
                isinstance(s, np.ndarray) and
                isinstance(ns, np.ndarray) and
                s.size > 0 and
                ns.size > 0 and
                not np.any(np.isnan(s)) and
                not np.any(np.isnan(ns))
            ):
                valid_batch.append(trans)

        if len(valid_batch) == 0:
            return

        state_batch, action_batch, reward_batch, next_state_batch = zip(*valid_batch)

        state_batch = torch.FloatTensor(np.array(state_batch))
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))

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

class Agent(DQNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)

    def state_to_vector(self, obs):
        agent_pos = obs["agent_pos"]  # (2,)
        # agent_angle = np.array([obs["agent_angle"]])  # (1,)
        # sensor_distances = np.array(obs["sensor_distances"])  # (3,)
        # pickup_point = obs["pickup_point"]  # (2,)
        has_order = np.array([float(obs["has_order"])])  # (1,)

        #One hot encoding of target table
        num_tables = len(obs["target_tables"])
        target_idx = None
        for idx, table in enumerate(obs["target_tables"]):
            if np.allclose(table, obs["current_target_table"], atol=1e-2):
                target_idx = idx
                break

        target_onehot = np.zeros(num_tables)
        if target_idx is not None:
            target_onehot[target_idx] = 1.0

        return np.concatenate([
            agent_pos,
            has_order,
            target_onehot
        ])

    def take_action(self, observation):
        state_vector = self.state_to_vector(observation)
        return self.select_action(state_vector)

    def observe_transition(self, state, action, reward, next_state, done):
        state_vector = self.state_to_vector(state)
        next_state_vector = self.state_to_vector(next_state)

        valid = all([
            state_vector is not None,
            next_state_vector is not None,
            not np.any(np.isnan(state_vector)),
            not np.any(np.isnan(next_state_vector)),
            state_vector.size > 0,
            next_state_vector.size > 0
        ])

        if valid:
            self.remember(state_vector, action, reward, next_state_vector)
            self.train()
        else:
            print("WARNING: Invalid transition skipped")

        # Save trained model to disk
        def save_model(self, filename):
            torch.save(self.model_net.state_dict(), filename)
            print(f"Model saved to {filename}")

        # Load trained model from disk
        def load_model(self, filename):
            self.model_net.load_state_dict(torch.load(filename))
            self.model_net.eval()  # Important! Set to evaluation mode
            print(f"Model loaded from {filename}")

