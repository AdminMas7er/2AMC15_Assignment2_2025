import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from .replaybuffer import ReplayBuffer  # relative import
from world.delivery_environment import DeliveryEnvironment

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

        # Filter out any remaining invalid transitions
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

        # Debug check — show bad states
        for i, s in enumerate(state_batch):
            if s is None or np.any(np.isnan(s)):
                print(f"WARNING: Bad state at index {i}: {s}")
        for i, s in enumerate(next_state_batch):
            if s is None or np.any(np.isnan(s)):
                print(f"WARNING: Bad next_state at index {i}: {s}")

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

    def take_action(self, observation):
        return self.select_action(observation)

    def observe_transition(self, state, action, reward, next_state, done):
        valid = all([
            state is not None,
            next_state is not None,
            isinstance(state, np.ndarray),
            isinstance(next_state, np.ndarray),
            not np.any(np.isnan(state)),
            not np.any(np.isnan(next_state)),
            state.size > 0,
            next_state.size > 0
        ])

        if valid:
            self.remember(state, action, reward, next_state)
            self.train()
        else:
            print("WARNING: Invalid transition skipped")


        # if done:
        #     self.reset_episode_state()

    # def reset_episode_state(self):
    #     self.memory.buffer = [(None, None, None, None)] * self.memory.size
    #     self.memory.index = 0
    #     self.memory.currentSize = 0
    #     print("INFO: ReplayBuffer cleared on episode reset.")
