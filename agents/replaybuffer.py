from random import sample

class ReplayBuffer:

    def __init__(self, size: int, batch_size :int):
        self.size = size
        self.batch_size = batch_size
        self.index = 0
        self.buffer = [(None, None, None, None)]*size
        self.currentSize = 0

    def store(self, state, action, reward, next_state):
        print(self.index)
        print(self.buffer)
        self.buffer[self.index] = (state, action, reward, next_state)
        self.index = (self.index + 1) % self.size
        self.currentSize += 1

    def sample(self):
        if self.currentSize < self.batch_size:
            raise ValueError("Buffer does not contain enough elements yet")
        return sample(self.buffer, self.batch_size)
    
