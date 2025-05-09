from collections import deque
import torch
import numpy as np

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
            self.priorities[idx] = abs(td_error.item()) + self.epsilon

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
        states_tensor = torch.tensor(states, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor,
                dones_tensor, indices, weights_tensor)


buffer = PrioritizedReplayBuffer(capacity=10)
PrioritizedReplayBuffer.sample = 3
print("Sampled transitions:\n", buffer.sample(3))