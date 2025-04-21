import gymnasium as gym
import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
import numpy as np

global Q
env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8", render_mode="human")
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = [np.zeros((num_states, num_actions)) for _ in range(2)]
# Q = [np.random.rand(8,4), np.random.rand(8,4)] 
learning_rate = 0.1
gamma = 0.99
num_episodes = 100


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


def update_q_tables(state, action, reward, next_state, done,truncated):
  	# Get the index of the table to update
    i = np.random.randint(0, 2)

    if done or truncated:
        # If the episode is done, update the Q-table with the reward
        Q[i][state, action] = (1 - learning_rate) * Q[i][state, action] + learning_rate * reward
        #print("I'm done")
    else:
        best_next_action = np.argmax(Q[i][next_state])
        Q[i][state, action] = (
            (1 - learning_rate) * Q[i][state, action]
            + learning_rate * (reward + gamma * Q[1 - i][next_state, best_next_action])
        )
       

if __name__ == "__main__":

    for episode in range(num_episodes):
        
        print("Episode: ", episode)

        state, info = env.reset()
        terminated = False   
        truncated = False

        while not (terminated or truncated):

            action = np.random.choice(num_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Update the Q-tables
            update_q_tables(state, action, reward, next_state,terminated,truncated)
            state = next_state
            
           
    # Combine the learned Q-tables    
    Q_combined = Q[0] + Q[1]
    policy = {state: np.argmax(Q_combined[state]) for state in range(num_states)}
    print("Final Policy: ", policy)
    #render_policy(policy)