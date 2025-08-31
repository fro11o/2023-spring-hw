from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import ale_py

gym.register_envs(ale_py)

# Training hyperparameters
learning_rate = 0.01  # How fast to learn (higher = faster but less stable)
# n_episodes = 100_000  # Number of hands to practice
n_episodes = 1_000  # Number of hands to practice
start_epsilon = 1.0  # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1  # Always keep some exploration


class BreakoutAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """
        Initialize a Q-Learning agent.

        Args:
            env: The training environment.
            learning_rate: How quickly to update Q-values (0-1).
            initial_epsilon: Starting exploration rate (usually 1.0).
            epsilon_decay: How much to reduce epsilon each episode.
            final_epsilon: Minimum exploration rate (usually 0.1).
            discount_factor: How much to value future rewards (0-1).
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: bytes) -> int:
        """
        Choose an action using epsilon-greedy strategy.
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: bytes,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: bytes,
    ):
        """
        Update Q-value based on experience.
        """
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


env_train = gym.make("ALE/Breakout-v5")
env_train = gym.wrappers.RecordEpisodeStatistics(env_train, buffer_length=n_episodes)

env_test = gym.make("ALE/Breakout-v5", render_mode="human")


agent = BreakoutAgent(
    env=env_train,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def train(env: gym.Env):
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        obs = obs.tobytes()  # Convert to bytes for hashing
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = next_obs.tobytes()  # Convert to bytes for hashing
            agent.update(obs,
                         action,
                         reward,
                         terminated,
                         next_obs)
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()


def visualize(env: gym.Env):
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(
        range(len(training_error_moving_average)),
        training_error_moving_average
    )
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


def test_agent(env: gym.Env, num_episodes=1):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        obs = obs.tobytes()  # Convert to bytes for hashing
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            obs = obs.tobytes()  # Convert to bytes for hashing
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    agent.epsilon = old_epsilon

    # win_rate = np.mean(np.array(total_rewards) > 0)
    # average_reward = np.mean(total_rewards)

    # print(f"Test Results over {num_episodes} episodes:")
    # print(f"Win Rate: {win_rate:.1%}")
    # print(f"Average Reward: {average_reward:.3f}")
    # print(f"Standard Deviation: {np.std(total_rewards):.3f}")


def main():
    for i in range(1):
        train(env_train)
        test_agent(env_test)
    visualize(env_train)


if __name__ == "__main__":
    main()