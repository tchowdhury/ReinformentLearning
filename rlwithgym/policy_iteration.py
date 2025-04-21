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

# Complete the render function
def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()

@lru_cache(maxsize=None)
def compute_state_value(state, policy_items, depth=0, max_depth=100):
    policy_dict = dict(policy_items)
       
    if state == terminal_state or depth >= max_depth:
        return 0
    
    action = policy_dict[state]
    _, next_state, reward, _ = env.unwrapped.P[state][action][0]
    return reward + gamma * compute_state_value(next_state ,policy_items, depth + 1)


# Complete the function to compute the action-value for a state-action pair
def compute_q_value(state, action, policy_items):
    
    if state == terminal_state:
        return None   
    probability, next_state, reward, done = env.unwrapped.P[state][action][0]
    return reward + gamma * compute_state_value(next_state, policy_items) 


# Complete the policy evaluation function
def policy_evaluation(policy):
    policy_items = tuple(policy.items())
    V = {state: compute_state_value(state,policy_items) for state in range(number_of_states)}
    compute_state_value.cache_clear()
    return V


def policy_improvement(policy):
    improved_policy = {s: 0 for s in range(number_of_states-1)}
    policy_items = tuple(policy.items())

	# Compute the Q-value for each state-action pair
    Q = {(state, action): compute_q_value(state,action,policy_items) for state in range(number_of_states) for action in range(number_of_actions)}
            
    # Compute the new policy based on the Q-values
    for state in range(number_of_states-1):
        max_action = max(range(number_of_actions), key=lambda action: Q[(state, action)])
        improved_policy[state] = max_action
        
    return improved_policy

# Complete the policy iteration function
def policy_iteration():
    policy = {0:1, 1:2, 2:1,3:0,4:1,5:2,6:1,7:1,8:2,9:1,10:1,11:1,12:1,13:2,14:2,15:0}
    while True:
        V = policy_evaluation(policy)
        improved_policy = policy_improvement(policy)
        if improved_policy == policy:
            break
        policy = improved_policy
    
    return policy, V


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

policy, V = policy_iteration()
render_policy(policy)