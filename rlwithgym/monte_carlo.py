import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')
number_of_states = env.observation_space.n
number_of_actions = env.action_space.n
gamma = 1  # Discount factor


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


def get_policy(Q):
    policy = {state: np.argmax(Q[state]) for state in range(number_of_states)}
    return policy


def generate_episode():    
    episode = []    
    state, info = env.reset(seed=42)    
    terminated = False
    while not terminated:        
        action = env.action_space.sample()        
        next_state, reward, terminated, truncated, info = env.step(action)        
        episode.append((state, action, reward))        
        state = next_state
    return episode    


def first_visit_mc(num_episodes):
    Q = np.zeros((number_of_states, number_of_actions))    
    returns_sum = np.zeros((number_of_states, number_of_actions))    
    returns_count = np.zeros((number_of_states, number_of_actions))

    for i in range(num_episodes):
        episode = generate_episode()
        visited_states = set()
        for j, (state, action, reward) in enumerate(episode):
            # Define the first-visit condition
            if (state, action) not in visited_states:
                # Update the returns, their counts and the visited states
                returns_sum[state, action] += sum([x[2] for x in episode[j:]])
                returns_count[state, action] += 1
                visited_states.add((state, action))

    nonzero_counts = returns_count != 0
    Q[nonzero_counts] = returns_sum[nonzero_counts] / returns_count[nonzero_counts]

    return Q


def every_visit_mc(num_episodes):
    Q = np.zeros((number_of_states, number_of_actions))    
    returns_sum = np.zeros((number_of_states, number_of_actions))    
    returns_count = np.zeros((number_of_states, number_of_actions))

    for i in range(num_episodes):
        episode = generate_episode()
        for j, (state, action, reward) in enumerate(episode):
            # Update the returns, their counts 
            returns_sum[state, action] += sum([x[2] for x in episode[j:]])
            returns_count[state, action] += 1

    nonzero_counts = returns_count != 0
    Q[nonzero_counts] = returns_sum[nonzero_counts] / returns_count[nonzero_counts]

    return Q

if __name__ == "__main__":
    q  = every_visit_mc(1000)
    policy = get_policy(q)
    print("Policy:")
    print(policy)
    render_policy(get_policy(q))
    
