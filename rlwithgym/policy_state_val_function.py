import gymnasium as gym
import matplotlib.pyplot as plt
from functools import lru_cache


# Defining a deterministic policy

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='rgb_array')

# Get the initial state
initial_state, info = env.reset()

# Complete the render function
def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()

terminal_state = 15  # Define the terminal state    
number_of_states = env.observation_space.n
#number_of_states = 16  # Number of states in FrozenLake
gamma = 0.9  # Discount factor

# Define the policy
policy = {0:1, 1:2, 2:1,3:0,4:1,5:2,6:1,7:1,8:2,9:1,10:1,11:1,12:1,13:2,14:2,15:0}



# Functon to compute the state value function
@lru_cache(maxsize=None)
def compute_state_value(state, depth=0, max_depth=100):
    print(f"Computing value for state {state}")
    
    if state == terminal_state or depth >= max_depth:
        return 0
    
    action = policy[state]
    _, next_state, reward, _ = env.unwrapped.P[state][action][0]
    return reward + gamma * compute_state_value(next_state, depth + 1)


# for state in range(number_of_states):
#     print(f"State {state}: Action {policy[state]}")
#     print(f"reward {compute_state_value(state)}")


# Compute all state values 
state_values = {state: compute_state_value(state) for state in range(number_of_states)}
print(state_values)


terminated = False
while not terminated:
  # Select action based on policy 
  action = policy[initial_state]
  state, reward, terminated, truncated, info = env.step(action)
  
  # Render the environment
  render()
#env.close()




