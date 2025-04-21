import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8", render_mode="rgb_array")
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))
learning_rate = 0.1
gamma = 1
num_episodes = 1000


def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.axis('off')
    plt.show()


def render_policy(policy):
  state, info = env.reset()
  terminated = False
  render()
  while not terminated:
    # Select action based on policy 
    action = policy[state]
    state, reward, terminated, truncated, info = env.step(action)
    # Render the environment
    render()


def get_policy():
    policy = {state: np.argmax(Q[state]) for state in range(num_states)}
    return policy


def update_q_table(state, action, reward, next_state, next_action):
  	# Get the old value of the current state-action pair
    old_value = Q[state, action]
    # Get the value of the next state-action pair
    next_value = Q[next_state, next_action]
    # Compute the new value of the current state-action pair
    Q[state, action] = (1 - learning_rate) * old_value + learning_rate * (reward + gamma * next_value)


if __name__ == "__main__":
    for episode in range(num_episodes):
        state, info = env.reset(seed=42)
        action = env.action_space.sample()
        terminated = False
        while not terminated:
            # Execute the action
            next_state, reward, terminated, truncated, info = env.step(action)
            # Choose the next action randomly
            next_action = env.action_space.sample()
            # Update the Q-table
            update_q_table(state, action, reward, next_state, next_action)
            state, action = next_state, next_action   
    
    render_policy(get_policy())