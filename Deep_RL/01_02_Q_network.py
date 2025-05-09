import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Implement a Q-network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Define a linear transformation layer
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        # Ensure the ReLU activation function is used
        x = torch.relu(self.fc1(torch.tensor(state, dtype=torch.float32)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
# Instantiate the Q-network for input dimension 8 and output dimension 4
state_size = 8  # Example state size for LunarLander-v3
action_size = 4  # Example action size for LunarLander-v3
q_network = QNetwork(state_size, action_size)

# Specify the optimizer learning rate
# and the optimizer
learning_rate = 0.0001  # Example learning rate
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

print("Q-Network initialized as:\n", q_network)  