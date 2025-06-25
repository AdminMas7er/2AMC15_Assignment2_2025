import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from agents.replaybuffer import ReplayBuffer
from pathlib import Path

def kaiming_init(m: nn.Module):
    """He‑uniform init for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.constant_(m.bias, 0)

class DQN_Network(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.apply(kaiming_init)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        map_width: float,
        map_height: float,
        target_update_freq: int = 1000,
        gamma: float = 0.99,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        learning_rate: float = 1e-3,
        device: torch.device | None = None,
        seed: int = 42,
        num_tables: int | None = None,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = device if device is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.state_size = state_size
        self.action_size = action_size
        self.num_tables  = (num_tables if num_tables is not None
                            else state_size - 4)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_network = DQN_Network(state_size, action_size).to(self.device)
        self.target_network = DQN_Network(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.train_step = 0
        self.map_range = np.array([map_width, map_height], dtype=np.float32)
        
    def state_to_vector(self, obs: dict | None):
        if obs is None:  # terminal placeholder
            return np.zeros(self.state_size, dtype=np.float32)
    
        # 1) position (x, y)  ->  0-1 range
        pos = np.asarray(obs["agent_pos"], dtype=np.float32)
        pos_norm = pos / self.map_range           # ← NEW
    
        # 2) heading (cosθ, sinθ) already in [-1, 1]
        heading = np.asarray(obs["heading"], dtype=np.float32)
    
        # 3) one-hot target id
        onehot = np.zeros(self.num_tables, dtype=np.float32)
        tid = int(obs["current_target"])
        if 0 <= tid < self.num_tables:
            onehot[tid] = 1.0
    
        return np.concatenate([pos_norm, heading, onehot]).astype(np.float32)



    # -------------------------------------------
    #  epsilon‑greedy action
    # -------------------------------------------
    def action(self, state: dict):
        eps = self.epsilon  # use current epsilon

        if random.random() < eps:
            return random.randrange(self.action_size)
        state_v = torch.as_tensor(self.state_to_vector(state), device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_network(state_v)
        return int(q_vals.argmax(dim=1).item())

    # -------------------------------------------
    #  store transition & train
    # -------------------------------------------
    def observe(self, s, a, r, s2, done):
        self.replay_buffer.store(
            self.state_to_vector(s),
            a,
            r,
            self.state_to_vector(s2),
            float(done),
        )
        self.optimize()

    # -------------------------------------------
    #  optimize single batch
    # -------------------------------------------
    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return
    
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*self.replay_buffer.sample())
        )
        s  = torch.as_tensor(states,       device=self.device)
        a  = torch.as_tensor(actions,      device=self.device).unsqueeze(1)
        r  = torch.as_tensor(rewards,      device=self.device).unsqueeze(1)
        s2 = torch.as_tensor(next_states,  device=self.device)
        d  = torch.as_tensor(dones,        device=self.device).unsqueeze(1)
    
        # --------------------- TD target & loss ---------------------------
        q_curr = self.q_network(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_network(s2).max(1)[0].unsqueeze(1)
            q_tgt  = r + self.gamma * q_next * (1 - d)
    
        loss = F.smooth_l1_loss(q_curr, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
    
        # --------------------- bookkeeping -------------------------------
        self.last_loss = loss.item()                    
    
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    # -------------------------------------------
    #  save / load
    # -------------------------------------------
    def save(self, path: str | Path):
        p = Path(path)
        if p.is_dir():
            p = p / "dqn_model.pth"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q": self.q_network.state_dict(),
            "target": self.target_network.state_dict(),
            "optim": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step": self.train_step,
        }, p)
        print(f"Model saved to {p}")

    def load(self, path: str | Path):
        p = Path(path)
        if p.is_dir():
            p = p / "dqn_model.pth"
        if not p.exists():
            print(f"[WARN] No checkpoint at {p}")
            return
        ckpt = torch.load(p, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q"])
        self.target_network.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.train_step = ckpt.get("step", 0)
        self.q_network.eval(); self.target_network.eval()
        print(f"Model loaded from {p}")

=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from agents.replaybuffer import ReplayBuffer
from pathlib import Path

def kaiming_init(m: nn.Module):
    """He‑uniform init for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.constant_(m.bias, 0)

class DQN_Network(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.apply(kaiming_init)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        map_width: float,
        map_height: float,
        target_update_freq: int = 1000,
        gamma: float = 0.99,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        learning_rate: float = 1e-3,
        device: torch.device | None = None,
        seed: int = 42,
        num_tables: int | None = None,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = device if device is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.state_size = state_size
        self.action_size = action_size
        self.num_tables  = (num_tables if num_tables is not None
                            else state_size - 4)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_network = DQN_Network(state_size, action_size).to(self.device)
        self.target_network = DQN_Network(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.train_step = 0
        self.map_range = np.array([map_width, map_height], dtype=np.float32)
        
    def state_to_vector(self, obs: dict | None):
        if obs is None:  # terminal placeholder
            return np.zeros(self.state_size, dtype=np.float32)
    
        # 1) position (x, y)  ->  0-1 range
        pos = np.asarray(obs["agent_pos"], dtype=np.float32)
        pos_norm = pos / self.map_range           # ← NEW
    
        # 2) heading (cosθ, sinθ) already in [-1, 1]
        heading = np.asarray(obs["heading"], dtype=np.float32)
    
        # 3) one-hot target id
        onehot = np.zeros(self.num_tables, dtype=np.float32)
        tid = int(obs["current_target"])
        if 0 <= tid < self.num_tables:
            onehot[tid] = 1.0
    
        return np.concatenate([pos_norm, heading, onehot]).astype(np.float32)



    # -------------------------------------------
    #  epsilon‑greedy action
    # -------------------------------------------
    def action(self, state: dict):
        eps = self.epsilon  # use current epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if random.random() < eps:
            return random.randrange(self.action_size)
        state_v = torch.as_tensor(self.state_to_vector(state), device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_network(state_v)
        return int(q_vals.argmax(dim=1).item())

    # -------------------------------------------
    #  store transition & train
    # -------------------------------------------
    def observe(self, s, a, r, s2, done):
        self.replay_buffer.store(
            self.state_to_vector(s),
            a,
            r,
            self.state_to_vector(s2),
            float(done),
        )
        self.optimize()

    # -------------------------------------------
    #  optimize single batch
    # -------------------------------------------
    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return
    
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*self.replay_buffer.sample())
        )
        s  = torch.as_tensor(states,       device=self.device)
        a  = torch.as_tensor(actions,      device=self.device).unsqueeze(1)
        r  = torch.as_tensor(rewards,      device=self.device).unsqueeze(1)
        s2 = torch.as_tensor(next_states,  device=self.device)
        d  = torch.as_tensor(dones,        device=self.device).unsqueeze(1)
    
        # --------------------- TD target & loss ---------------------------
        q_curr = self.q_network(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_network(s2).max(1)[0].unsqueeze(1)
            q_tgt  = r + self.gamma * q_next * (1 - d)
    
        loss = F.smooth_l1_loss(q_curr, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
    
        # --------------------- bookkeeping -------------------------------
        self.last_loss = loss.item()                    
    
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    # -------------------------------------------
    #  save / load
    # -------------------------------------------
    def save(self, path: str | Path):
        p = Path(path)
        if p.is_dir():
            p = p / "dqn_model.pth"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q": self.q_network.state_dict(),
            "target": self.target_network.state_dict(),
            "optim": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step": self.train_step,
        }, p)
        print(f"Model saved to {p}")

    def load(self, path: str | Path):
        p = Path(path)
        if p.is_dir():
            p = p / "dqn_model.pth"
        if not p.exists():
            print(f"[WARN] No checkpoint at {p}")
            return
        ckpt = torch.load(p, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q"])
        self.target_network.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.train_step = ckpt.get("step", 0)
        self.q_network.eval(); self.target_network.eval()
        print(f"Model loaded from {p}")

>>>>>>> 3e3742c9c5dcb7d58e652ba8fe58df4c994fc3c0
