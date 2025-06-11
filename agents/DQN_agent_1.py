import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """Simple 3-layer MLP mapping state → Q-values for each action."""
    def __init__(self, state_size, action_size, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)


class DQNAgent:
    """Interacts with and learns from the environment using DQN."""
    def __init__(
        self,
        state_size,
        action_size,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update=1000,
        device=None
    ):
        # Device selection
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Networks
        self.policy_net = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        # Initialize target_net with same weights as policy_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.action_size = action_size

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Internal counters
        self.step_count = 0

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        :param state: np.ndarray of shape (state_size,)
        :return: action index (int)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def observe_transition(self, state, action, reward, next_state, done):
        """
        Save experience and trigger learning step.
        :param state, action, reward, next_state, done: single transition
        """
        self.memory.add(state, action, reward, next_state, done)
        self.learn()
        
    def decay_epsilon(self):
        """Decay epsilon at the end of an episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self):
        """
        Sample a batch from memory and perform a single Q-learning update.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # Convert to tensors
        states      = torch.from_numpy(states).float().to(self.device)
        actions     = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards     = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones       = torch.from_numpy(dones.astype(np.uint8)).float().unsqueeze(1).to(self.device)

        # Compute current Q values
        q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q   = rewards + (1 - dones) * self.gamma * max_next_q

        # Compute loss
        loss = self.loss_fn(q_values, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network params
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        """Save the policy network weights to disk."""
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath):
        """Load policy network weights and sync to target network."""
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
