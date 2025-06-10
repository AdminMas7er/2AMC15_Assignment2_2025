from random import sample

class ReplayBuffer:

    def __init__(self, size: int, batch_size: int, minimal_experience: int):
        self.size = size
        self.batch_size = batch_size
        self.index = 0
        self.buffer = [(None, None, None, None)]*size
        self.currentSize = 0
        self.minimal_experience = minimal_experience if minimal_experience < size else size

    def store(self, state, action, reward, next_state, done):
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.size
        self.currentSize += 1

    def sample(self):
        if self.currentSize < self.minimal_experience:
            raise ValueError("Buffer does not contain enough elements yet")
        return sample(self.buffer, self.batch_size)
    
