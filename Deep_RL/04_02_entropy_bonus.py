import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP error

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import gymnasium as gym

# Hyperparameters
gamma = 0.99  # Discount factor for future rewards
learning_rate = 0.0001  # Learning rate for the optimizer
num_episodes = 1000  # Total number of episodes for training
batch_size = 64  # Batch size for experience replay
tau = 0.005  # Soft update parameter for target network
c_entropy = 0.01  # Coefficient for the entropy bonus
epsilon = .2  # Epsilon for PPO clipping

# Initiate the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")  

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

# Test the CriticNetwork class
critic_network = Critic(8)

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
    #print(f"+{'-'*29}+\n|         Ratio: {str(ratio)} |\n| Clipped ratio: {str(clipped_ratio)} |\n+{'-'*29}+\n")
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

state_size = 8  # Example state size for LunarLander-v3
action_size = 4  # Example action size for LunarLander-v3
actor = ActorNetwork(state_size, action_size)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
critic = Critic(state_size)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)


def select_action(actor_network, state):
  # Obtain the action probabilities
  action_probs = actor_network(state)
  #print('Action probabilities:', action_probs)
  # Instantiate the action distribution
  action_dist = Categorical(action_probs)
  # Sample an action from the distribution
  action = action_dist.sample()
  log_prob = action_dist.log_prob(action)
  # Obtain the entropy of the policy  
  entropy = action_dist.entropy()
  return (action.item(), log_prob.reshape(1), entropy)



def describe_episode(episode, reward, episode_reward, t):
    landed = reward == 100
    crashed = reward == -100
    print(
        f"| Episode {episode+1:4} | Duration: {t:4} steps | Return: {episode_reward:<7.2f} |",
        "Landed   |" if landed else "Crashed  |" if crashed else "Hovering |",
    )  

def plot_probabilities(probs):
    dist = Categorical(torch.tensor(probs))
    entropy = dist.entropy() / math.log(2)  # entropy in bits
    print(f"{'Probabilities:':>15} {[round(prob, 3) for prob in dist.probs.tolist()]}")
    print(f"{'Entropy:':>15} {entropy.item():.2f}\n")

    plt.figure()
    plt.bar([str(x) for x in range(len(dist.probs))], dist.probs.tolist(), color='skyblue', edgecolor='black')
    plt.ylabel('Probability')
    plt.xlabel('Action index')
    plt.ylim(0, 1)
    plt.show()

# Try with different distributions
# plot_probabilities([.25, .25, .25, .25])
# plot_probabilities([.1, .15, .2, .25, .3])
# plot_probabilities([.15, .35, .12, .25, .13])

actor_losses = torch.tensor([])
critic_losses = torch.tensor([])

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0
    step = 0
    while not done:    
        step += 1
        action, action_log_prob, entropy = select_action(actor, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        actor_loss, critic_losses = calculate_losses(critic, action_log_prob, action_log_prob,
                                                   reward, state, next_state, done)
        
        # Append to the loss tensors
        actor_losses = torch.cat((actor_losses, actor_loss))
        critic_losses = torch.cat((critic_losses, critic_losses))

        if len(actor_losses) >= 10:
            # Calculate the batch losses
            # Calculate the mean of the last 10 losses
            actor_loss_batch = actor_losses.mean()
            critic_loss_batch = critic_losses.mean()

            # Remove the entropy bonus from the actor loss
            #actor_loss -= c_entropy * entropy
            actor_optimizer.zero_grad(); actor_loss_batch.backward(); actor_optimizer.step()
            critic_optimizer.zero_grad(); critic_loss_batch.backward(); critic_optimizer.step()
            # Reinitialize the loss tensors
            actor_losses = torch.tensor([])
            critic_losses = torch.tensor([])
            state = next_state
    describe_episode(episode, reward, episode_reward, step)