import gym
import numpy as np
import torch
import time
from ddpg import DDPGAgent  # Assumes the converted PyTorch DDPGAgent
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 2000
MAX_STEPS = 1000
RENDER = False

SAVE_PREFIX = "lunarlander_ddpg"
NOISE_SCALE = 0.1  # Initial noise scale for exploration
NOISE_DECAY = 0.999  # Noise decay per episode
MIN_NOISE_SCALE = 0.01  # Minimum noise scale


def train_lunarlander():
    # Initialize the Lunar Lander environment
    env = gym.make("LunarLanderContinuous-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # Initialize the DDPG agent
    agent = DDPGAgent(state_dim, action_dim, action_bound)

    # Metrics for plotting and analysis
    rewards_history = []
    moving_avg_rewards = []
    loss_history = []  # To track per-episode loss
    moving_avg_losses = []  # 100-episode moving average for loss
    start_time = time.time()

    noise_scale = NOISE_SCALE

    for episode in range(EPISODES):
        # Reset environment and extract state
        state = env.reset()
        if isinstance(state, tuple):  # Check for tuple return
            state = state[0]
        state = np.array(state, dtype=np.float32)  # Ensure state is a NumPy array

        episode_reward = 0
        episode_loss = 0  # Accumulate loss for the episode

        for step in range(MAX_STEPS):
            if RENDER:
                env.render()

            # Take action using noisy policy for exploration
            action = agent.noisy_policy(state, noise_scale)

            # Clip and format the action properly
            action = np.clip(action, -action_bound, action_bound).astype(np.float32)

            # Step the environment
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Ensure next_state is properly formatted
            if isinstance(next_state, tuple):  # Check for tuple return
                next_state = next_state[0]
            next_state = np.array(next_state, dtype=np.float32)  # Convert to NumPy array

            # Add to replay buffer
            agent.buffer.add(state, action, reward, next_state, done)

            losses = agent.update()
            if losses is not None:
                critic_loss, actor_loss = losses
                episode_loss += critic_loss  # Only accumulate critic loss for tracking
                
            # Transition to the next state
            state = next_state
            episode_reward += reward

            if done:
                break

        # Decay noise for the next episode
        noise_scale = max(MIN_NOISE_SCALE, noise_scale * NOISE_DECAY)

        # Log rewards and losses
        rewards_history.append(episode_reward)
        loss_history.append(episode_loss)
        moving_avg_rewards.append(np.mean(rewards_history[-100:]))  # 100-episode moving average
        moving_avg_losses.append(np.mean(loss_history[-100:]))  # 100-episode moving average for loss

        print(f"Episode {episode + 1}/{EPISODES}, Reward: {episode_reward:.2f}, "
              f"Loss: {episode_loss:.2f}, "
              f"Average Rewards in Last 100 episodes: {moving_avg_rewards[-1]:.2f}")

        # Termination criteria
        if moving_avg_rewards[-1] >= 200.0 and episode >= 100:
            print(f"Environment solved in {episode + 1} episodes!")
            break

    # Save trained model
    agent.save_model(SAVE_PREFIX)
    
    # Save training metrics
    np.save(f"{SAVE_PREFIX}_rewards.npy", rewards_history)
    np.save(f"{SAVE_PREFIX}_moving_avg_rewards.npy", moving_avg_rewards)
    np.save(f"{SAVE_PREFIX}_losses.npy", loss_history)
    np.save(f"{SAVE_PREFIX}_moving_avg_losses.npy", moving_avg_losses)

    # Report training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    with open('training_time.txt', 'w') as f:
        f.write(f"Total time taken to train the agent: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds\n")

    # Plot training progress
    plot_training_progress(rewards_history, moving_avg_rewards, loss_history, moving_avg_losses)

    env.close()


def plot_training_progress(rewards, moving_avg_rewards, losses, moving_avg_losses):

    # Plot rewards and moving average rewards
    plt.figure(figsize=(12, 8))

    # Rewards subplot
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label="Episode Reward")
    plt.plot(moving_avg_rewards, label="100-Episode Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress - Rewards")
    plt.legend()
    plt.grid()

    # Losses subplot
    plt.subplot(2, 1, 2)
    plt.plot(losses, label="Episode Loss")
    plt.plot(moving_avg_losses, label="100-Episode Moving Avg Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Progress - Losses")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{SAVE_PREFIX}_training_progress.png")
    plt.show()


if __name__ == "__main__":
    train_lunarlander()
