import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8", render_mode="human")
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))
#Q = np.random.rand(5, 2)
#print("Old Q:\n", Q)
learning_rate = 0.1
gamma = 0.99
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


def update_q_table(state, action, next_state, reward):
  	# Calculate the expected Q-value for the next state
    expected_q = np.mean(Q[next_state])
    # Update the Q-value for the current state and action
    Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate * (reward + gamma * expected_q)



if __name__ == "__main__":
    for i_episode in range(num_episodes):
        state, info = env.reset()    
        done = False    
        while not done: 
            action = env.action_space.sample()               
            next_state, reward, done, truncated, info = env.step(action)
            # Update the Q-table
            update_q_table(state, action, next_state, reward)
            state = next_state
    # Derive policy from Q-table        
    #render_policy(get_policy())