import gymnasium as gym
import matplotlib.pyplot as plt


# Create the environment
env = gym.make("MountainCar-v0", render_mode="human")

# Get the initial state
initial_state, info = env.reset(seed=42)

position = initial_state[0]
velocity = initial_state[1]

print(f"The position of the car along the x-axis is {position} (m)")
print(f"The velocity of the car is {velocity} (m/s)")

# Complete the render function
def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()

# Call the render function    
#render()

# Define the sequence of actions
actions = [0,1,2,0,1,2,0,1,2,0,1,2,2,0,1,2,0,1,2,0,1,2,0,1,2] 
for action in actions:
    # Take the action
    state, reward, terminated, truncated, info = env.step(action)
    
    # Print the state and reward
    print(f"State: {state}, Reward: {reward}")
    
    # Render the environment
    #render()

    if terminated:
        print("You reached the goal!")