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
# state_value = critic_network(torch.rand(8))
# print('State value:', state_value)  

def calculate_losses(critic_network, action_log_prob, 
                     reward, state, next_state, done):
    value = critic_network(state)
    next_value = critic_network(next_state)
    # Calculate the TD target
    td_target = (reward + gamma * next_value * (1-done))
    td_error = td_target - value
    # Calculate the actor loss
    actor_loss = -action_log_prob * td_error.detach()
    # Calculate the critic loss
    critic_loss = td_error ** 2
    return actor_loss, critic_loss
  
# Test the calculate_losses function
# state = torch.rand(8)
# action_log_prob = torch.rand(1)
# reward = torch.rand(1)
# next_state = torch.rand(8)
# done = torch.tensor([0.0])
# actor_network = ActorNetwork(8, 4)
# actor = actor_network(state)
# action = torch.argmax(actor).item()
# action_log_prob = torch.log(actor[action])
# actor_loss, critic_loss = calculate_losses(critic_network, action_log_prob, 
#                                              reward, state, next_state, done)

# print(round(actor_loss.item(), 2), round(critic_loss.item(), 2))

# Instantiate the Q-network for input dimension 8 and output dimension 4
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
  return action.item(), log_prob.reshape(1)


def describe_episode(episode, reward, episode_reward, t):
    landed = reward == 100
    crashed = reward == -100
    print(
        f"| Episode {episode+1:4} | Duration: {t:4} steps | Return: {episode_reward:<7.2f} |",
        "Landed   |" if landed else "Crashed  |" if crashed else "Hovering |",
    )  

# Train the actor and critic networks
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0
    step = 0
    while not done:
        step += 1
        if done:
            break
        # Select the action
        action, action_log_prob = select_action(actor, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        # Calculate the losses
        actor_loss, critic_loss = calculate_losses(critic, action_log_prob, reward, state, next_state, done)        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        state = next_state

    describe_episode(episode, reward, episode_reward, step)