from collections import deque
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

gamma = 0.99  # Discount factor for future rewards

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
        x = torch.relu(self.fc1(torch.tensor(state)))
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


# Define a function to describe the episode
# This function prints the episode number, duration, return, and status (landed/crashed/hovering)
def describe_episode(episode, reward, episode_reward, t):
    landed = reward == 100
    crashed = reward == -100
    print(
        f"| Episode {episode+1:4} | Duration: {t:4} steps | Return: {episode_reward:<7.2f} |",
        "Landed   |" if landed else "Crashed  |" if crashed else "Hovering |",
    )    

replay_buffer = ReplayBuffer(10000)  # Initialize the replay buffer with a capacity of 10,000
batch_size = 64  # Batch size for experience replay    

for episode in range(50):
    state, info = env.reset()
    done = False
    step = 0
    episode_reward = 0
    while not done:
        step += 1
        q_values = q_network(state)        
        action = torch.argmax(q_values).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Store the latest experience in the replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
        if len(replay_buffer) >= batch_size:
            # Sample 64 experiences from the replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            q_values = q_network(states).gather(1, actions).squeeze(1)
            # Obtain the next state Q-values
            next_state_q_values = q_network(next_states).amax(1)
            target_q_values = rewards + gamma * next_state_q_values * (1-dones)
            loss = nn.MSELoss()(target_q_values, q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       
        state = next_state
        episode_reward += reward    
    describe_episode(episode, reward, episode_reward, step)