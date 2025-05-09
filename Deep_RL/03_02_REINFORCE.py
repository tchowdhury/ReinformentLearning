import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym


# Hyperparameters
gamma = 0.99  # Discount factor for future rewards
learning_rate = 0.0001  # Learning rate for the optimizer
num_episodes = 1000  # Total number of episodes for training
batch_size = 64  # Batch size for experience replay
tau = 0.005  # Soft update parameter for target network

# Initiate the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")  

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


def sample_from_distribution(probs):
    print(f"\nInput: {probs}")
    probs = torch.tensor(probs, dtype=torch.float32)
    # Instantiate the categorical distribution
    dist = Categorical(probs)
    # Take one sample from the distribution
    sampled_index = dist.sample()
    print(f"Taking one sample: index {sampled_index}, with associated probability {dist.probs[sampled_index]:.2f}")  


def select_action(policy_network, state):
  # Obtain the action probabilities
  action_probs = policy_network(state)
  #print('Action probabilities:', action_probs)
  # Instantiate the action distribution
  action_dist = Categorical(action_probs)
  # Sample an action from the distribution
  action = action_dist.sample()
  log_prob = action_dist.log_prob(action)
  return action.item(), log_prob.reshape(1)    

# Test the select_action function
# policy_network = PolicyNetwork(8, 4)    
# state = torch.rand(8)
# action, log_prob = select_action(policy_network, state)
# print('Sampled action index:', action)
# print(f'Log probability of sampled action: {log_prob.item():.2f}')

# Define a function to describe the episode
# This function prints the episode number, duration, return, and status (landed/crashed/hovering)
def describe_episode(episode, reward, episode_reward, t):
    landed = reward == 100
    crashed = reward == -100
    print(
        f"| Episode {episode+1:4} | Duration: {t:4} steps | Return: {episode_reward:<7.2f} |",
        "Landed   |" if landed else "Crashed  |" if crashed else "Hovering |",
    )  

# Instantiate the Q-network for input dimension 8 and output dimension 4
state_size = 8  # Example state size for LunarLander-v3
action_size = 4  # Example action size for LunarLander-v3
policy_network = PolicyNetwork(state_size, action_size)
optimizer = torch.optim.Adam(policy_network.parameters(), lr=learning_rate)

# REINFORCE training loop
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0
    step = 0
    episode_log_probs = torch.tensor([])
    R = 0
    while not done:
        step += 1
        action, log_prob = select_action(policy_network, state)                
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        # Append to the episode action log probabilities
        episode_log_probs = torch.cat((episode_log_probs, log_prob))
        # Increment the episode return
        R += (gamma  ** step) * reward
        state = next_state
    # Calculate the episode loss
    loss = R * episode_log_probs.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    describe_episode(episode, reward, episode_reward, step)