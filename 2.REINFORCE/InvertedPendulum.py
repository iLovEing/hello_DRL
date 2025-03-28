import os
import torch
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import gymnasium as gym
from loguru import logger
from torch.distributions.normal import Normal
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


ENV_NAME = 'InvertedPendulum-v5'
LOG_FILE = f'{ENV_NAME}.txt'
SEED = 1111

num_episodes = 10000
discount_factor = 0.99
learning_rate = 1e-4
stop_reward = 950


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        output = self.fc(x)
        mean, std = torch.split(output, 1, dim=-1)
        std = torch.log(1 + torch.exp(std))
        return mean, std

class Agent:
    def __init__(self, ckpt_path=None):
        self.policy = PolicyNet(4, 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load(ckpt_path)
        self.policy = self.policy.to(self.device)
        self.train_mode(False)

    def act(self, state: np.array):
        if self.training:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action_mean, action_std = self.policy.forward(state)
            distrib = Normal(action_mean[0], action_std[0])
            action = distrib.sample()
            log_prob = distrib.log_prob(action)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action_mean, action_std = self.policy.forward(state)
                distrib = Normal(action_mean[0], action_std[0])
                action = action_mean[0]
                log_prob = distrib.log_prob(action)

        return action, log_prob

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        if path is not None:
            self.policy.load_state_dict(torch.load(path, device=self.device))

    def train_mode(self, training=True):
        self.training = training
        if training:
            self.policy.train()
        else:
            self.policy.eval()


class REINFORCE:
    def __init__(self, agent, gamma=0.99, lr=1e-4):
        self.agent = agent
        self.gamma = gamma
        self.agent.train_mode()
        self.optimizer = torch.optim.Adam(self.agent.policy.parameters(), lr=lr)

        self.log_probs = []
        self.rewards = []

    def select_action(self, obs, env=None):
        return self.agent.act(obs)

    def update(self, log_prob, reward, done):
        if not done:
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            return

        returns = []
        for _r in self.rewards[::-1]:
            returns.insert(0, _r + (self.gamma * returns[-1] if len(returns) > 0 else 0))

        log_probs = torch.stack(self.log_probs)
        returns = torch.tensor(returns).to(log_probs.device)
        loss = -(log_probs * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()


def save_video(ckpt=None):
    num_eval_episodes = 4

    env = gym.make(ENV_NAME, render_mode="rgb_array")  # replace with your environment
    env = RecordVideo(env, video_folder="vedio", name_prefix=ENV_NAME,
                      episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
    agent = Agent(load_ckpt=ckpt)

    for episode_num in range(num_eval_episodes):
        obs, info = env.reset()

        episode_over = False
        while not episode_over:
            action, _ = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())

            episode_over = terminated or truncated
    env.close()

    print(f'Episode time taken: {env.time_queue}')
    print(f'Episode total rewards: {env.return_queue}')
    print(f'Episode lengths: {env.length_queue}')


def play_video(ckpt=None):
    env = gym.make(ENV_NAME, render_mode="human")
    agent = Agent(load_ckpt=ckpt)

    observation, info = env.reset()
    episode_over = False
    while not episode_over:
        action, _ = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())

        episode_over = terminated or truncated

    env.close()


def test():
    if False:
    # if True:
        play_video(ckpt=os.path.join('ckpt', 'InvertedPendulum-v5_episode_6979_reward_955.pth'))
    else:
        save_video(ckpt=os.path.join('ckpt', 'InvertedPendulum-v5_episode_6979_reward_955.pth'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train():
    set_seed(SEED)

    env = gym.make(ENV_NAME)
    env = gym.wrappers.RecordEpisodeStatistics(env, 50)
    agent = Agent()
    algo = REINFORCE(agent, gamma=discount_factor, lr=learning_rate)

    logger.info(f'Training agent to play {ENV_NAME} by REINFORCE.')
    logger.info(f'num_episodes:{num_episodes}, '
                f'discount_rate:{discount_factor}, '
                f'learning_rate:{learning_rate}, '
                f'seed:{SEED}')

    cumulative_rewards = []
    avg_rewards = []
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()

        done = False
        while not done:
            action, log_prob = algo.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            done = terminated or truncated
            algo.update(log_prob, reward, done)

        avg_reward = int(np.mean(env.return_queue))
        cumulative_rewards.append(int(env.return_queue[-1]))
        avg_rewards.append(avg_reward)

        if (episode + 1) % 50 == 0:
            logger.info(f"Episode:{episode + 1}, Average Reward:{avg_reward}")

        if avg_reward > stop_reward or episode + 1 == num_episodes:
            logger.info(f'training finished at episode {episode + 1}, average reward: {avg_reward}')
            agent.save(os.path.join('ckpt', f'{ENV_NAME}_episode_{episode + 1}_reward_{avg_reward}.pth'))
            break

    data_df = pd.DataFrame({
        'episode': range(1, len(cumulative_rewards) + 1, 1),
        'reward': cumulative_rewards,
        'avg_reward': avg_rewards,
    })

    plt.figure(dpi=200)
    sns.set(rc={"figure.figsize": (20, 6)})
    fig = sns.lineplot(x='episode', y='value', hue='variable',
                       data=pd.melt(data_df, ['episode']), palette=['blue', 'red'])
    plt.show()
    scatter_fig = fig.get_figure()
    scatter_fig.savefig(os.path.join('train_log', f'{ENV_NAME}_reward.png'), dpi=400)


if __name__ == '__main__':
    logger.add(os.path.join('train_log', f'{LOG_FILE}'),
               format="{time:HH:mm:ss.SSS} | {file}:{line} | {level} | {message}")
    # train()
    test()
