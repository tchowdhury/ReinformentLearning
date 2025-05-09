import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
  def __init__(self, state_size, action_size):
    super(PolicyNetwork, self).__init__()
    self.fc1 = nn.Linear(state_size, 64)
    self.fc2 = nn.Linear(64, 64)
    # Give the desired size for the output layer
    self.fc3 = nn.Linear(64, action_size)

  def forward(self, state):
    x = torch.relu(self.fc1(torch.FloatTensor(state).clone().detach()))
    x = torch.relu(self.fc2(x))
    # Obtain the action probabilities
    action_probs = torch.softmax(self.fc3(x), dim=-1)
    return action_probs

# Test the PolicyNetwork class
# state = torch.tensor([0.93, 0.95, 0.93, 0.32, 0.60, 0.65, 0.93, 0.58])
# policy_network = PolicyNetwork(8, 4)
# action_probs = policy_network(state)
# print('Action probabilities:', action_probs)

def sample_from_distribution(probs):
    print(f"\nInput: {probs}")
    probs = torch.tensor(probs, dtype=torch.float32)
    # Instantiate the categorical distribution
    dist = Categorical(probs)
    # Take one sample from the distribution
    sampled_index = dist.sample()
    print(f"Taking one sample: index {sampled_index}, with associated probability {dist.probs[sampled_index]:.2f}")

# Test the sample_from_distribution function
# # Specify 3 positive numbers summing to 1
# sample_from_distribution([.3, .5, .2])
# # Specify 5 positive numbers that do not sum to 1
# sample_from_distribution([0.2, 0.3, 0.1, 0.1, 0.2])