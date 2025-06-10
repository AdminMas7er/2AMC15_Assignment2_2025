"""Soft Actor-Critic Agent."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from replaybuffer import ReplayBuffer
import numpy as np
from agents import BaseAgent

class QNetwork(nn.Module):
    """
    The critic

    Takes the state as input and outputs the estimated q-values for all actions.
    We use 2 hidden layers with dimensions of 256 each (or 'hidden_dimension'), as by https://arxiv.org/pdf/1812.05905
    
    
    
    """
    def __init__(self, input_dimension, output_dimension, hidden_dimension=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dimension, hidden_dimension) # 'Fully-connected layer 1'
        self.fc2 = nn.Linear(hidden_dimension, hidden_dimension) # 'Fully-connected layer 2'
        self.fc3 = nn.Linear(hidden_dimension, output_dimension) # 'Fully-connected layer 3'
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.fc3(x) # The raw numerial output of layer 3 equals estimation of the q-values for all actions

class PolicyNetwork(nn.Module):
    """
    The actor

    Takes the state as input and outputs the probabilities of all actions.
    """
    def __init__(self, input_dimension, output_dimension, hidden_dimension=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.fc3 = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x) # The raw output of layer 3 equals the logits for the actions, that we still need to transform into probabilities
        probabilities = F.softmax(logits, dim=-1) # Softmax to turn the logits into actual probabilities for each action
        log_probabilities = F.log_softmax(logits, dim=-1) # Log probabilities for the actions, needed for the loss function
        return probabilities, log_probabilities
    
    
class SACAgent(BaseAgent):
    def __init__(
            self,
            state_dimension,
            action_dimension,
            hidden_dimension=256,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256,
            target_entropy=None

    ):
        super(SACAgent, self).__init__()
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        self.hidden_dimension = hidden_dimension
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer=ReplayBuffer(size=buffer_size, batch_size=batch_size, minimal_experience=10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pass
    def take_action():
        pass
    def update():
        pass



"""
# Code skeleton/pseudocode

Algo 1 from the paper:

Init parameter vectors psi, psi_, tau and phi
(representing the neural network approximations of , respectively, the value critic V, target value network V, Q function critic Q and a stochastic policy)
for each iteration:
    for each environment step:
        a <- sampled from policy pi_phi, based on current state
        s_new <- execute the action (aka get the new state based on the trans probability)
        add the state, action, reward and new state to the replay buffer
    
    for each gradient step, update parameter vectors:
        sample minibatch from the replay buffer
        
        
        """