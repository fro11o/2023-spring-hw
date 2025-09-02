from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import ale_py
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import argparse

gym.register_envs(ale_py)

# Training hyperparameters
learning_rate = 1e-4 # How fast to learn (higher = faster but less stable)
# n_episodes = 100_000  # Number of hands to practice
n_episodes = 3_000  # Number of hands to practice
start_epsilon = 0.1  # Start with 100% random actions
epsilon_decay = start_epsilon / n_episodes  # Reduce exploration over time
final_epsilon = 0.1  # Always keep some exploration


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 25 * 19, 512),  # <-- fix output size here!
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, buffer_size=10000, batch_size=32, gamma=0.99):
        self.env = env
        obs_shape = (3, 210, 160)
        self.n_actions = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(obs_shape, self.n_actions).to(self.device)
        self.target_net = DQN(obs_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.steps = 0

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(obs)
        # print(q_values)
        return int(torch.argmax(q_values).item())

    def store_transition(self, *args):
        self.memory.append(Transition(*args))

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        return Transition(*zip(*batch))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.sample_batch()
        state_batch = torch.tensor(np.array(transitions.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(transitions.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(np.array(transitions.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(transitions.done, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, target)
        # print(q_values)
        # print(target)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_error.append(loss.item())

        # Update target network
        if self.steps % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


env_train = gym.make("ALE/Breakout-v5")
env_train = gym.wrappers.RecordEpisodeStatistics(env_train, buffer_length=n_episodes)

env_test = gym.make("ALE/Breakout-v5", render_mode="human")


agent = DQNAgent(
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


def preprocess_obs(obs):
    # Normalize to [0,1]
    obs = obs.astype(np.float32) / 255.0
    return obs.transpose(2, 0, 1)  # shape: (3, 210, 160)


def train(env: gym.Env):
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        obs = preprocess_obs(obs)
        done = False
        episode_reward = 0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = preprocess_obs(next_obs)
            done_flag = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done_flag)
            agent.update()
            obs = next_obs
            done = done_flag
            episode_reward += reward
        print(f"Episode Reward: {episode_reward}")
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
        obs = preprocess_obs(obs)
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = preprocess_obs(next_obs)
            obs = next_obs
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


def save_agent(agent, filename="dqn_breakout.pth"):
    torch.save(agent.policy_net.state_dict(), filename)

def load_agent(agent, filename="dqn_breakout.pth"):
    agent.policy_net.load_state_dict(torch.load(filename))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Only evaluate the agent, do not train')
    parser.add_argument('--init-weights', type=str, default=None, help='Initialize agent weights from file')
    args = parser.parse_args()

    if args.init_weights:
        load_agent(agent, args.init_weights)

    if args.eval:
        test_agent(env_test, num_episodes=5)
    else:
        train(env_train)
        save_agent(agent)
        test_agent(env_test, num_episodes=5)
        visualize(env_train)

if __name__ == "__main__":
    main()