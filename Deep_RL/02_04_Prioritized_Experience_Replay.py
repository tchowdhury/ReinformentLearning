from collections import deque
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math

# Hyperparameters
gamma = 0.99  # Discount factor for future rewards
learning_rate = 0.0001  # Learning rate for the optimizer
num_episodes = 1000  # Total number of episodes for training
batch_size = 64  # Batch size for experience replay
tau = 0.005  # Soft update parameter for target network

# Initiate the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")   

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
        # Ensure the state is converted to a tensor properly
        x = torch.relu(self.fc1(torch.FloatTensor(state).clone().detach()))  # Convert input to tensor properly
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def select_action(q_values, step, start, end, decay):
    
    # Calculate the threshold value for this step
    epsilon = end + (start - end) * math.exp(-step / decay)
    # Draw a random number between 0 and 1
    sample = random.random()
    if sample < epsilon:
        # Return a random action index
        return random.choice(range(len(q_values))) # Random action
    return torch.argmax(q_values).item() # Best action based on Q-values
      
# Test the select_action function
# for step in [1, 500, 2500]:
#     actions = [select_action(torch.Tensor([1, 2, 3, 5]), step, .9, .05, 1000) for _ in range(20)]
#     print(f"Selecting 20 actions at step {step}.\nThe action with highest q-value is action 3.\nSelected actions: {actions}\n\n")

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience_tuple = (state, action, reward, next_state, done)
        # Append experience_tuple to the memory buffer
        self.memory.append(experience_tuple)    
    
    def __len__(self):
        return len(self.memory)
    
    def sample(self, batch_size):
        # Draw a random sample of size batch_size
        batch = random.sample(self.memory, batch_size)
        # Transform batch into a tuple of lists
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.tensor(states, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        # Ensure actions_tensor has shape (batch_size, 1)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor


# A prioritized replay buffer that uses a deque to store the transitions
class PrioritizedReplayBuffer:
    def __init__(
        self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01
    ):
        self.memory = deque(maxlen=capacity)
        self.alpha, self.beta, self.beta_increment, self.epsilon = (alpha, beta, beta_increment, epsilon)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience_tuple = (state, action, reward, next_state, done)
        # Initialize the transition's priority
        max_priority = max(self.priorities) if self.memory else 1.0
        self.memory.append(experience_tuple)
        self.priorities.append(max_priority)
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors.tolist()):
            # Update the transition's priority
            self.priorities[idx] = abs(td_error) + self.epsilon

    def increase_beta(self):
        # Increase beta if less than 1
        self.beta = min(1.0, self.beta + self.beta_increment)

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        # Calculate the sampling probabilities
        probabilities = priorities**self.alpha / np.sum(priorities**self.alpha)
        # Draw the indices for the sample
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        # Calculate the importance weights
        weights = (1 / (len(self.memory) * probabilities)) ** self.beta
        weights /= np.max(weights)
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])
        weights = [weights[idx] for idx in indices]
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor,
                dones_tensor, indices, weights_tensor)

# Test the PrioritizedReplayBuffer class      
# buffer = PrioritizedReplayBuffer(capacity=3)
# buffer.push(state=[1,3], action=2, reward=1, next_state=[2,4], done=False)
# print("Transition in memory buffer:", buffer.memory)
# print("Priority buffer:", buffer.priorities)

def update_target_network(target_network, online_network, tau):
    # Obtain the state dicts for both networks
    target_net_state_dict = target_network.state_dict()
    online_net_state_dict = online_network.state_dict()
    for key in online_net_state_dict:
        # Calculate the updated state dict for the target network
        target_net_state_dict[key] = (online_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau))
        # Load the updated state dict into the target network
        target_network.load_state_dict(target_net_state_dict)
    return None

# Instantiate the Q-network for input dimension 8 and output dimension 4
state_size = 8  # Example state size for LunarLander-v3
action_size = 4  # Example action size for LunarLander-v3
online_network = QNetwork(state_size, action_size)  # Example state size for LunarLander-v3
target_network = QNetwork(state_size, action_size)  # Example state size for LunarLander-v3
#q_network = QNetwork(state_size, action_size)



# Test the update_target_network function

# print("online network weights:", print_state_dict(online_network))
# print("target network weights (pre-update):", print_state_dict(target_network))
# update_target_network(target_network, online_network, .001)
# print("target network weights (post-update):", print_state_dict(target_network))

optimizer = optim.Adam(online_network.parameters(), lr=learning_rate)

# Function to calculate the state-action value function
def print_state_dict(network):
  sdict = network.state_dict()
  result_str = '\n------------------------------------\n'
  for key in sdict:
    result_str += f'layer {key}:\n'
    result_str += '\n'.join(['\t' + x for x in sdict[key].__str__().split('\n')]) + '\n'
  result_str += '------------------------------------\n\n'
  return result_str


# Define a function to describe the episode
# This function prints the episode number, duration, return, and status (landed/crashed/hovering)
def describe_episode(episode, reward, episode_reward, t):
    landed = reward == 100
    crashed = reward == -100
    print(
        f"| Episode {episode+1:4} | Duration: {t:4} steps | Return: {episode_reward:<7.2f} |",
        "Landed   |" if landed else "Crashed  |" if crashed else "Hovering |",
    )    

#replay_buffer = ReplayBuffer(10000)  # Initialize the replay buffer with a capacity of 10,000
total_steps = 0  # Initialize the total number of steps taken 
# Instantiate a Prioritized Replay Buffer with capacity 10000
replay_buffer = PrioritizedReplayBuffer(10000)

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    step = 0
    episode_reward = 0
    # Increase the replay buffer's beta parameter
    replay_buffer.increase_beta()
    while not done:
        step += 1
        total_steps += 1
        q_values = online_network(state)
        # Select the action with epsilon greediness
        action = select_action(q_values, total_steps, start=.9, end=.05, decay=1000)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)        
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(64)
            q_values = online_network(states).gather(1, actions).squeeze(1)
            # Ensure gradients are not tracked
            with torch.no_grad():
                next_q_values = target_network(next_states).amax(1)
                target_q_values = rewards + gamma * next_q_values * (1-dones)            
            
            td_errors = target_q_values - q_values
            #print(f"Type of td_errors: {type(td_errors)}")
            
            # Update the replay buffer priorities for that batch
            replay_buffer.update_priorities(indices, td_errors)
            loss = torch.sum(weights * (q_values - target_q_values) ** 2)
            #loss = nn.MSELoss()(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update the target network weights
            update_target_network(target_network, online_network, tau=.005)
        state = next_state
        episode_reward += reward    
    describe_episode(episode, reward, episode_reward, step)