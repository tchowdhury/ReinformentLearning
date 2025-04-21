import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8", render_mode="rgb_array")
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))
learning_rate = 0.1
gamma = 0.99
num_episodes = 1000
rewards_per_episode = []
avg_reward_per_random_episode = 0.0021



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
        total_reward = 0
        terminated = False
        while not terminated:
            #action = env.action_space.sample()
            action = np.argmax(Q[state])
            # Execute the action
            next_state, reward, terminated, truncated, info = env.step(action)
            # Update the Q-table
            update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            #print(reward)
        # Append the total reward to the rewards list    
        rewards_per_episode.append(total_reward)

# Compute and print the average reward per learned episode
avg_reward_per_learned_episode = np.mean(rewards_per_episode)
print("Average reward per learned episode: ", avg_reward_per_learned_episode)
print("Average reward per random episode: ", avg_reward_per_random_episode)