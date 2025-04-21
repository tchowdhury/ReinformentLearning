import gymnasium as gym
import matplotlib.pyplot as plt
from functools import lru_cache


env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')
terminal_state = 15  # Define the terminal state    
number_of_states = env.observation_space.n
number_of_actions = env.action_space.n  # Number of actions in FrozenLake
gamma = 0.9  # Discount factor

V = {state: 0 for state in range(number_of_states)}
policy = {state:0 for state in range(number_of_states-1)}
threshold = 0.001

# Get the initial state
initial_state, info = env.reset()

# Complete the render function
def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()


# @lru_cache(maxsize=None)
# def compute_state_value(state, policy_items, depth=0, max_depth=100):
#     policy_dict = dict(policy_items)
       
#     if state == terminal_state or depth >= max_depth:
#         return 0
    
#     action = policy_dict[state]
#     _, next_state, reward, _ = env.unwrapped.P[state][action][0]
#     return reward + gamma * compute_state_value(next_state ,policy_items, depth + 1)


# Complete the function to compute the action-value for a state-action pair
def compute_q_value(state, action, V):
    if state == terminal_state:
        return None
    _, next_state, reward, _ = env.unwrapped.P[state][action][0]
    return reward + gamma * V[next_state]


def get_max_action_and_value(state, V):
  
  Q_values = [compute_q_value(state, action, V) for action in range(number_of_actions)]
  max_action = max(range(number_of_actions), key=lambda a: Q_values[a])
  max_q_value = Q_values[max_action]
  return max_action, max_q_value

def render_policy(policy):
  state, info = env.reset()
  render()
  terminated = False
  while not terminated:
    # Select action based on policy 
    action = policy[state]
    state, reward, terminated, truncated, info = env.step(action)
    # Render the environment
    render()


while True:
  new_V = {state: 0 for state in range(number_of_states)}
  
  for state in range(number_of_states-1):
    # Get action with maximum Q-value and its value 
    max_action, max_q_value = get_max_action_and_value(state, V)
    # Update the value function and policy
    new_V[state] = max_q_value
    policy[state] = max_action
  
  # Test if change in state values is negligeable
  if all(abs(new_V[state] - V[state]) < threshold for state in V):  
    break
  V = new_V

render_policy(policy)    