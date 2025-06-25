import random
from collections import deque

class ReplayBuffer:
    def __init__(self, maxlen: int, batch_size: int):
        self.batch_size = batch_size
        self.samples = 0
        self.buffer = deque([], maxlen=maxlen)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state,done))
        self.samples += 1

    def sample(self):
        if len(self.buffer) < self.batch_size:
            return None
        return random.sample(self.buffer, self.batch_size)
    def __len__(self):
        return len(self.buffer)
    