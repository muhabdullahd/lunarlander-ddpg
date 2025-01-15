import gym
import numpy as np
from ddpg import DDPGAgent


MAX_STEPS = 1000 
NOISE_SCALE = 0.1  # Initial noise scale for exploration
NOISE_DECAY = 0.999  # Slower decay
MIN_NOISE_SCALE = 0.01  # Slightly higher minimum noise


# Define the test function
def test_lunarlander():
    """
    Test the trained DDPG agent on the LunarLanderContinuous-v2 environment.
    The environment renders the lunar lander attempting to land on the moon surface.
    """

    # Load the LunarLanderContinuous-v2 environment
    env = gym.make("LunarLanderContinuous-v2", render_mode="human") 
    state_dim = env.observation_space.shape[0]  # Get state dimension
    action_dim = env.action_space.shape[0]  # Get action dimension
    action_bound = env.action_space.high[0]  # Get the action bounds

    # Initialize the DDPG agent
    agent = DDPGAgent(state_dim, action_dim, action_bound)

    # Load the pre-trained weights into the agent
    agent.load_model("lunarlander_ddpg")

    noise_scale = NOISE_SCALE

    total_reward = 0

    # Run the testing phase for 10 episodes
    NUM_TEST_EPISODES = 10
    for episode in range(NUM_TEST_EPISODES):
        print(f"Starting Episode {episode + 1}")

        # Reset the environment to get the initial state
        state = env.reset()
        if isinstance(state, tuple):  # Check if Gym returns a tuple
            state = state[0]
        state = np.array(state, dtype=np.float32)  # Convert state to NumPy array

        episode_reward = 0  # Initialize reward counter for the episode

        for step in range(MAX_STEPS):  # Explicitly limit the episode to MAX_STEPS
            # Render the environment for visualization
            env.render()

            # Use the agent's noisy policy for testing (small noise for exploration)
            # action = agent.noisy_policy(state, noise_scale)
            action = agent.policy(state)

            # Clip and format the action to ensure it adheres to the environment's limits
            action = np.clip(action, -action_bound, action_bound).astype(np.float32)

            # Step the environment with the chosen action
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated  # Handle Gym's transition flags

            # Ensure next_state is properly formatted
            if isinstance(next_state, tuple):  # Check for tuple return
                next_state = next_state[0]
            next_state = np.array(next_state, dtype=np.float32)  # Convert to NumPy array

            state = next_state

            # Accumulate the reward
            episode_reward += reward

            # Break the loop if the episode ends
            if done:
                print(f"Episode {episode + 1} ended after {step + 1} steps with reward: {episode_reward:.2f}")
                total_reward += episode_reward
                break
        # Decay noise for the next episode
        noise_scale = max(MIN_NOISE_SCALE, noise_scale * NOISE_DECAY)
    print("avg: ", total_reward/10)
        
    # Close the environment after testing
    env.close()

if __name__ == "__main__":
    # Entry point for the testing script
    test_lunarlander()

