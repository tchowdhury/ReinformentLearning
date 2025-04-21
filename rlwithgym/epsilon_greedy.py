import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8", render_mode="human")
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))
learning_rate = 0.1
gamma = 0.99
num_episodes = 100
rewards_per_episode = []
epsilon = 0.9
rewards_eps_greedy = []
max_steps = 100
min_epsilon = 0.1
decay_rate = 0.099


def epsilon_greedy(state):
    # Implement the condition to explore
    if np.random.rand() < epsilon:
      	# Choose a random action
        action = env.action_space.sample() # Explore
    else:
      	# Choose the best action according to q_table
        action = np.argmax(Q[state, :]) # Exploit
    return action


def update_q_table(state, action, reward, next_state):
  	# Get the old value of the current state-action pair
    old_value = Q[state, action]
    # Determine the maximum Q-value for the next state
    next_max = max(Q[next_state])
    # Compute the new value of the current state-action pair
    Q[state, action] = (1 - learning_rate) * old_value + learning_rate * (reward + gamma * next_max)


if __name__ == "__main__":

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        
        for i in range(max_steps):
            # Select action with epsilon-greedy strategy
            action = epsilon_greedy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            # Accumulate reward
            episode_reward += reward        
            update_q_table(state, action, reward, next_state)      
            state = next_state
        
        # Append the toal reward to the rewards list 
        rewards_eps_greedy.append(episode_reward)
        # Update epsilon
        epsilon = max(min_epsilon, epsilon * (1 - decay_rate))

    # Compute and print the average reward per learned episode
    
    print("Average reward per episode: ", np.mean(rewards_eps_greedy))