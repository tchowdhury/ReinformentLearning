import gymnasium as gym
import matplotlib.pyplot as plt
from functools import lru_cache


env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')
terminal_state = 15  # Define the terminal state    
number_of_states = env.observation_space.n
number_of_actions = env.action_space.n  # Number of actions in FrozenLake
gamma = 0.9  # Discount factor

# Get the initial state
initial_state, info = env.reset()

# Define the policy
policy = {0:1, 1:2, 2:1,3:0,4:1,5:2,6:1,7:1,8:2,9:1,10:1,11:1,12:1,13:2,14:2,15:0}

# Complete the render function
def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()

@lru_cache(maxsize=None)
def compute_state_value(state, depth=0, max_depth=100):
    print(f"Computing value for state {state}")
    
    if state == terminal_state or depth >= max_depth:
        return 0
    
    action = policy[state]
    _, next_state, reward, _ = env.unwrapped.P[state][action][0]
    return reward + gamma * compute_state_value(next_state , depth + 1)

# Complete the function to compute the action-value for a state-action pair
def compute_q_value(state, action):
    if state == terminal_state:
        return None   
    probability, next_state, reward, done = env.unwrapped.P[state][action][0]
    return reward + gamma * compute_state_value(next_state)    

# Compute Q-values for each state-action pair
Q = {(state, action): compute_q_value(state,action) for state in range(number_of_states) for action in range(number_of_actions)}

print(Q)

# Improving policy using Q-values
improved_policy = {}

for state in range(number_of_states-1):
    # Find the best action for each state based on Q-values
    max_action = max(range(number_of_actions), key=lambda action: Q[(state, action)])
    improved_policy[state] = max_action

terminated = False
while not terminated:
  # Select action based on policy 
  action = improved_policy[state]
  # Execute the action
  state, reward, terminated, truncated, info = env.step(action)
  render()