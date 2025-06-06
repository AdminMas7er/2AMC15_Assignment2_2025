
import numpy as np
import random

class Agent:
    def __init__(self):
        self.max_speed = 0.2  # max forward movement per step
        self.max_turn = np.radians(30)  # max rotation per step

    def take_action(self, observation):
        # Random linear and angular velocities
        return random.randint(0,4)
