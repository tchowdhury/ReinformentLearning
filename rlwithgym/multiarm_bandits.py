import numpy as np
from matplotlib import pyplot as plt


n_iterations = 1000  # Number of iterations for the bandit problem
epsilon = 1.0  # Probability of exploration
min_epsilon = 0.01  # Minimum value of epsilon
epsilon_decay = 0.999  # Decay rate for epsilon
n_bandits = 10  # Number of bandits


def create_multi_armed_bandit(n_bandits):
  	# Generate the true bandits probabilities
    true_bandit_probs = np.random.rand(n_bandits) 
    # Create arrays that store the count and value for each bandit
    counts = np.zeros(n_bandits)  # How many times each bandit was played  
    values = np.zeros(n_bandits)  # Estimated winning probability of each bandit  
    # Create arrays that store the rewards and selected arms each episode
    rewards = np.zeros(n_iterations)  # Reward history
    selected_arms = np.zeros(n_iterations, dtype=int)  # Arm selection history 
    return true_bandit_probs, counts, values, rewards, selected_arms


def epsilon_greedy():
    """Selects an arm using the epsilon-greedy strategy."""
    if np.random.rand() < epsilon:
        return np.random.randint(len(values))  # Explore
    else:
        return np.argmax(values)  # Exploit


if __name__ == "__main__":
    # Create a 10-armed bandit
    true_bandit_probs, counts, values, rewards, selected_arms = create_multi_armed_bandit(n_bandits)

    for i in range(n_iterations): 
        # Select an arm
        arm = epsilon_greedy()
        # Compute the received reward
        reward = np.random.rand() < true_bandit_probs[arm]
        rewards[i] = reward
        selected_arms[i] = arm
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        # Update epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Initialize the selection percentages with zeros
    selections_percentage = np.zeros((n_iterations, n_bandits))
    # Fill the selection percentages with 1 for the selected arms
    # for each iteration
    for i in range(n_iterations):
        selections_percentage[i, selected_arms[i]] = 1
    
    # Compute the cumulative selection percentages 
    selections_percentage = np.cumsum(selections_percentage, axis=0) / np.arange(1, n_iterations + 1).reshape(-1, 1)
    
    for arm in range(n_bandits):
        # Plot the cumulative selection percentage for each arm
        plt.plot(selections_percentage[:, arm], label=f'Bandit #{arm+1}')
    
    plt.xlabel('Iteration Number')
    plt.ylabel('Percentage of Bandit Selections (%)')
    plt.legend()
    plt.show()
    
    for i, prob in enumerate(true_bandit_probs, 1):
        print(f"Bandit #{i} -> {prob:.2f}")    