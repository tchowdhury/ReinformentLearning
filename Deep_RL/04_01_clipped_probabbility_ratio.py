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
epsilon = .2  # Epsilon for PPO clipping

# log_prob = torch.tensor(.5).log()
# log_prob_old = torch.tensor(.4).log()


class ActorNetwork(nn.Module):
  def __init__(self, state_size, action_size):
    super(ActorNetwork, self).__init__()
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

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        # Fill in the desired dimensions
        self.fc2 = nn.Linear(64,1)

    def forward(self, state):
        x = torch.relu(self.fc1(torch.FloatTensor(state).clone().detach()))
        # Calculate the output value
        value = self.fc2(x)
        return value
    

def calculate_ratios(action_log_prob, action_log_prob_old, epsilon):
    # Obtain prob and prob_old
    prob = action_log_prob.exp()
    prob_old = action_log_prob_old.exp()
    
    # Detach the old action log prob
    prob_old_detached = prob_old.detach()

    # Calculate the probability ratio
    ratio = prob / prob_old_detached

    # Apply clipping
    clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    print(f"+{'-'*29}+\n|         Ratio: {str(ratio)} |\n| Clipped ratio: {str(clipped_ratio)} |\n+{'-'*29}+\n")
    return (ratio, clipped_ratio)


def calculate_losses(critic_network, action_log_prob, action_log_prob_old,
                     reward, state, next_state, done):
    value = critic_network(state)
    next_value = critic_network(next_state)
    td_target = (reward + gamma * next_value * (1-done))
    td_error = td_target - value

    # Obtain the probability ratios
    ratio, clipped_ratio = calculate_ratios(action_log_prob, action_log_prob_old, epsilon=.2)

    # Calculate the surrogate objectives
    surr1 = ratio * td_error.detach()
    surr2 = clipped_ratio * td_error.detach()    

    # Calculate the clipped surrogate objective
    objective = torch.min(surr1, surr2)

    # Calculate the actor loss
    actor_loss = -objective
    critic_loss = td_error ** 2
    return actor_loss, critic_loss

# Test the calculate_losses function
state = torch.rand(8)
action_log_prob = torch.rand(1)
reward = torch.rand(1)
next_state = torch.rand(8)
done = torch.tensor([0.0])
critic_network = Critic(8)
actor_network = ActorNetwork(8, 4)
actor = actor_network(state)
action = torch.argmax(actor).item()
action_log_prob = torch.log(actor[action])
action_log_prob_old = action_log_prob.clone().detach()
action_log_prob_old.requires_grad = False
# Calculate the losses
actor_loss, critic_loss = calculate_losses(critic_network, action_log_prob, 
                                             action_log_prob_old, reward, state, next_state, done)
# Print the losses
print(f"Actor Loss: {actor_loss.item()}")
print(f"Critic Loss: {critic_loss.item()}")
print(actor_loss, critic_loss)
