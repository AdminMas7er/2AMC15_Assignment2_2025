"""Soft Actor-Critic Agent."""

import numpy as np
from agents import BaseAgent

class SACAgent():
    def __init__():
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