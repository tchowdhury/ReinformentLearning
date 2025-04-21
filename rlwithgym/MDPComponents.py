import gymnasium as gym


def determine_environment_type_using_same_sequence_of_actions() -> tuple[list[int], list[int]]:
    """
    Determine if the environment is deterministic or stochastic by resetting to the same seed and taking the same sequence of actions.
    If the resulting observations differ, the environment is stochastic else deterministic.
    """
    env1 = gym.make('FrozenLake-v1', is_slippery=True)
    # Set up two identical episodes
    obs1, _ = env1.reset(seed=42)
    env1.action_space.seed(42)
    trajectory1 = []

    for _ in range(20):
        action = env1.action_space.sample()
        obs1, reward1, terminated1, truncated1, _ = env1.step(action)
        trajectory1.append(obs1)
        if terminated1 or truncated1:
            break


    env1.close()

    env2 = gym.make('FrozenLake-v1', is_slippery=True)
    # Repeat the exact same process
    obs2, _ = env2.reset(seed=42)
    env2.action_space.seed(42)
    trajectory2 = []

    for _ in range(20):
        action = env2.action_space.sample()
        obs2, reward2, terminated2, truncated2, _ = env2.step(action)
        trajectory2.append(obs2)
        if terminated2 or truncated2:
            break

    env2.close()

    return trajectory1 , trajectory2


def determine_environment_type_using_observation() -> list[int]:
    """
    Check if the environment is deterministic by replaying actions multiple times and checking if the resulting observations are identical.
    """
    env3 = gym.make('FrozenLake-v1', is_slippery=True)
    obs1, _ = env3.reset(seed=123)
    actions = [env3.action_space.sample() for _ in range(5)]

    # Replay actions multiple times
    results = []
    for _ in range(3):
        obs, _ = env3.reset(seed=123)
        temp = []
        for a in actions:
            obs, _, term, trunc, _ = env3.step(a)
            temp.append(obs)
            if term or trunc:
                break
        results.append(temp)

    env3.close()

    return results


def explore_state_and_action_spaces() -> tuple[int, int]:
    """
    Explore the state and action spaces of the FrozenLake environment.
    """
    # Create the Cliff Walking environment
    newenv = gym.make("CliffWalking-v0")

    # Compute the size of the action space
    num_actions = newenv.action_space.n
    
    # Compute the size of the state space
    num_states = newenv.observation_space.n

    newenv.close()
    
    return num_states, num_actions


def explore_transition_probabilities_and_rewards() -> dict:
    """
    Explore the transition probabilities and rewards of the FrozenLake environment.
    """
    # Create the Cliff Walking environment
    newenv = gym.make("CliffWalking-v0")

    # Choose the state
    state = newenv.reset()[0]
    print(f"State: {state}")

    # Compute the size of the action space
    num_actions = newenv.action_space.n

    # Extract transitions for each state-action pair
    for action in range(num_actions):
        transitions = newenv.unwrapped.P[state][action]

    newenv.close()

    return transitions    


def determine_number_of_terminal_states_for_discrte_environment() -> set:
    """
    Determine the number of terminal states in the FrozenLake environment.
    """
    import gymnasium as gym

    env = gym.make("FrozenLake-v1", is_slippery=True)
    unwrapped_env = env.unwrapped  # THIS is the raw env that has .P

    terminal_states = set()

    for state in range(unwrapped_env.observation_space.n):
        for action in range(unwrapped_env.action_space.n):
            for prob, next_state, reward, done in unwrapped_env.P[state][action]:
                if done:
                    terminal_states.add(next_state)
    
    env.close()

    return terminal_states


def determine_number_of_terminal_states_for_continuous_environment() -> set:
    """
    Determine the number of terminal states in the FrozenLake environment.
    """
    import gymnasium as gym

    env = gym.make("MountainCar-v0")
    unwrapped_env = env.unwrapped  # THIS is the raw env that has .P

    terminal_obs = set()

    for episode in range(1000):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                terminal_obs.add(tuple(obs))  # Approximate terminal states
        
    env.close()

    return terminal_obs


if __name__ == "__main__":

    print("Exploring the FrozenLake environment")
    print("="*50)

    trajectory1, trajectory2 = determine_environment_type_using_same_sequence_of_actions()
    # print("Trajectory 1:", trajectory1)
    # print("Trajectory 2:", trajectory2) 
    # Compare the two trajectories
    print("Is deterministic?", trajectory1 == trajectory2)
    results = determine_environment_type_using_observation()
    # Print the results
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}")
    print("Are all action sequences identical?", all(x == results[0] for x in results))

    print("Exploring state and action spaces")
    print("="*50)
    num_states, num_actions = explore_state_and_action_spaces()
    print(f"Number of states: {num_states}")
    print(f"Number of actions: {num_actions}")

    print("Exploring transition probabilities and rewards")
    print("="*50)
    transitions = explore_transition_probabilities_and_rewards()
    #print(f"Transitions for state {transitions}:")
    # Print details of each transition
    for transition in transitions:
        probability, next_state, reward, done = transition
        print(f"Probability: {probability}, Next State: {next_state}, Reward: {reward}, Done: {done}")

    print("Determine number of terminal states")
    print("="*50)
    terminal_states = determine_number_of_terminal_states_for_discrte_environment()
    print(f"Number of terminal states: {len(terminal_states)}")
    print(f"Terminal states: {terminal_states}")
    terminal_obs = determine_number_of_terminal_states_for_continuous_environment()
    print("Estimated number of terminal states (by unique terminal observations):", len(terminal_obs))