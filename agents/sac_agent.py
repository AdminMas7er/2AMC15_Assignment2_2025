"""Soft Actor-Critic Agent."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from replaybuffer import ReplayBuffer
import numpy as np
from agents import BaseAgent

class QNetwork(nn.Module):
    """
    The critic

    Takes the state as input and outputs the estimated q-values for all actions.
    We use 2 hidden layers with dimensions of 256 each (or 'hidden_dimension'), as by https://arxiv.org/pdf/1812.05905
    
    
    
    """
    def __init__(self, input_dimension: int, output_dimension: int, hidden_dimension: int = 256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dimension, hidden_dimension) # 'Fully-connected layer 1'
        self.fc2 = nn.Linear(hidden_dimension, hidden_dimension) # 'Fully-connected layer 2'
        self.fc3 = nn.Linear(hidden_dimension, output_dimension) # 'Fully-connected layer 3'
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.fc3(x) # The raw numerial output of layer 3 equals estimation of the q-values for all actions

        return q
    
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
                                                        # Axis =-1 because actions are the last dimension 

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
        self.state_dimension = state_dimension # Set dimensions
        self.action_dimension = action_dimension
        self.hidden_dimension = hidden_dimension

        self.gamma = gamma # Set hyperparamters
        self.tau = tau
        self.batch_size = batch_size 
        self.buffer=ReplayBuffer(size=buffer_size, batch_size=batch_size, minimal_experience=10000) # Create replay buffer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

        # Initialize networks
        self.Q1 = QNetwork(state_dimension,action_dimension, hidden_dimension).to(self.device) # Critic 1
        self.Q2 = QNetwork(state_dimension,action_dimension, hidden_dimension).to(self.device) # Critic 2
        self.Q1_target = QNetwork(state_dimension,action_dimension, hidden_dimension).to(self.device) # Target critic 1
        self.Q2_target = QNetwork(state_dimension,action_dimension, hidden_dimension).to(self.device) # Target critic 2 
        self.policy = PolicyNetwork(state_dimension, action_dimension, hidden_dimension).to(self.device) # Actor

        # Initialy sync the target networks with the main networks
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        # Adam Optimizers as used in https://arxiv.org/pdf/1910.07207
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        


    def take_action(self, state):
        """Sample an action from the policy for the current state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Convert state to tensor
        with torch.no_grad(): # Disabled for efficiency, as we do not need gradients for the action selection
            probabilities, log_probabilities = self.policy(state_tensor)
        distribution = Categorical(probabilities)  # Create a categorical distribution from the probabilities
        action = distribution.sample()  # Sample an action from the distribution
        
        return action.item() # Convert Tensor to Python int 
    
    def update(self, sampled_batch):
        """Update the agent's networks based on a sampled batch from the replay buffer."""

        # Unpack the sampled batch
        states, actions, rewards, next_states, dones = sampled_batch

        # And convert them to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ==Eq 4 from https://arxiv.org/pdf/1910.07207==
        # Update Qnetwork
        # Compute the target Q-values using the target networks
        with torch.no_grad():
            next_probabilities, next_log_probabilities = self.policy(next_states) # pi(a' | s')
            Q1_target_values = self.Q1_target(next_states) # Q_1'(s',.)
            Q2_target_values = self.Q2_target(next_states)  # Q_2'(s',.)
            min_Q_target = torch.min(Q1_target_values, Q2_target_values)  # Take the minimum of the two target Q-values ( min_i Q_i')

            # V(s') = sum_a pi(a|s') [ Q'(s',a) – α log pi(a|s') ]
            next_v = (next_probabilities * (min_Q_target - self.alpha * next_log_probabilities)).sum(dim=1, keepdim=True)
            Q_target = rewards + self.gamma * (1 - dones) * next_v  # Compute the target Q-values (y)

        current_Q1_values = self.Q1(states).gather(1, actions)  # Q_1(s,a)
        current_Q2_values = self.Q2(states).gather(1, actions)

        Q1_loss = F.mse_loss(current_Q1_values, Q_target)  # Compute the loss for Q_1
        Q2_loss = F.mse_loss(current_Q2_values, Q_target) # Compute the loss for Q_2

        # Update the Q-networks
        self.q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.q2_optimizer.step()
         
         # ==Eq 12 from https://arxiv.org/pdf/1910.07207==
        # Update the Policy network
        probabilities, log_probabilities = self.policy(states)
        Q1_values = self.Q1(states)
        Q2_values = self.Q2(states)

        min_Q_values = torch.min(Q1_values, Q2_values)  # Take the minimum of the two Q-values
        policy_loss = (probabilities * (self.alpha * log_probabilities - min_Q_values)).sum(dim=1, keepdim=True).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Eq 11 from https://arxiv.org/pdf/1910.07207
        # Update the temperature parameter (alpha)

        # TO BE EXPLAINED
        with torch.no_grad():
            entropy = - (probabilities * log_probabilities).sum(dim=1)
        alpha_loss = (self.log_alpha* (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update the target networks using soft update
        for parameters, target_parameters in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            target_parameters.data.copy_(self.tau * parameters.data + (1 - self.tau) * target_parameters.data)
        for parameters, target_parameters in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            target_parameters.data.copy_(self.tau * parameters.data + (1 - self.tau) * target_parameters.data)

        return {
            "q1_loss": Q1_loss.item(),
            "q2_loss": Q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha_value": self.alpha.item(),
        }




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