import os
import torch
import random
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import gymnasium as gym
from loguru import logger
from collections import deque
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


ENV_NAME = 'CartPole-v1-DDQN'
LOG_FILE = f'{ENV_NAME}.txt'
SEED = 1111
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005

num_episodes = 1000
batch_size = 128
replay_buffer_len = 10000
discount_factor = 0.99
learning_rate = 1e-4
stop_reward = 475


class QValueNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.fc(x)

class Agent:
    def __init__(self, ckpt_path=None):
        self.Q_net = QValueNet(4, 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load(ckpt_path=ckpt_path)
        self.Q_net = self.Q_net.to(self.device)
        self.train_mode(False)

    def act(self, state):
        assert isinstance(state, np.ndarray) or isinstance(state, torch.Tensor)
        if torch.is_tensor(state):
            state = state.to(self.device)
        else:
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)

        if self.training:
            action_values = self.Q_net.forward(state)
            action = torch.argmax(action_values, dim=-1)
            action_value = action_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        else:
            with torch.no_grad():
                action_values = self.Q_net.forward(state)
                action = torch.argmax(action_values, dim=-1)
                action_value = action_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action, action_value

    def save(self, path):
        torch.save(self.Q_net.state_dict(), path)

    def load(self, ckpt_path=None, state_dict=None, tau=1.):
        assert ckpt_path is None or state_dict is None, f'both ckpt_path and ckpt_state_dict are not None'

        if ckpt_path is not None:
            self.Q_net.load_state_dict(torch.load(ckpt_path))
        if state_dict is not None:
            temp_state_dict = self.Q_net.state_dict()
            for key in temp_state_dict.keys():
                temp_state_dict[key] = state_dict[key] * tau + temp_state_dict[key] * (1 - tau)
            self.Q_net.load_state_dict(temp_state_dict)

    def train_mode(self, training=True):
        self.training = training
        if training:
            self.Q_net.train()
        else:
            self.Q_net.eval()

class DQN:
    def __init__(self, agent, memory_size=1000, gamma=0.99, tau=0.005,
                 batch_size=128, lr=1e-4, loss_f=nn.SmoothL1Loss()):
        self.memory_size = memory_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.loss_f = loss_f
        self.act_decay_step = 0

        self.replay_buffer = deque(maxlen=memory_size)

        self.policy_agent = agent
        self.policy_agent.train_mode()
        self.optimizer = torch.optim.Adam(self.policy_agent.Q_net.parameters(), lr=lr)

        self.target_agent = Agent()
        self.refresh_target_agent()

    def refresh_target_agent(self, tau=1.0):
        self.target_agent.load(state_dict=self.policy_agent.Q_net.state_dict(), tau=tau)

    def select_action(self, env, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.act_decay_step / EPS_DECAY)
        self.act_decay_step += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action, _ = self.policy_agent.act(state)
                return action.item()
        else:
            return env.action_space.sample()

    def update(self, obs, action, reward, next_obs):
        self.replay_buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs
        })
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.stack([x['obs'] for x in batch]).to(self.policy_agent.device)
        action_batch = torch.stack([x['action'] for x in batch]).to(self.policy_agent.device)
        reward_batch = torch.stack([x['reward'] for x in batch]).to(self.target_agent.device)
        next_states = [x['next_obs'] for x in batch]

        output_qsa = self.policy_agent.Q_net(state_batch).gather(1, action_batch)

        # Compute the expected Q values
        non_final_mask = torch.tensor(list(map(lambda x: x is not None, next_states)), dtype=torch.bool)
        non_final_next_states = torch.stack(list(filter(lambda x: x is not None, next_states))).to(self.policy_agent.device)
        next_qsa = torch.zeros(self.batch_size, device=self.target_agent.device)
        with torch.no_grad():
            _q_value = self.policy_agent.Q_net(non_final_next_states)
            _q = torch.argmax(_q_value, dim=-1).to(self.target_agent.device)
            next_qsa[non_final_mask] = self.target_agent.Q_net(non_final_next_states).gather(1, _q.unsqueeze(-1)).squeeze(-1)
            expected_qsa = (next_qsa * self.gamma).unsqueeze(-1) + reward_batch

        loss = self.loss_f(output_qsa, expected_qsa)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_agent.Q_net.parameters(), 100)
        self.optimizer.step()

        self.refresh_target_agent(tau=self.tau)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train():
    set_seed(SEED)

    env = gym.make(ENV_NAME[:-5])
    env = gym.wrappers.RecordEpisodeStatistics(env, 50)
    agent = Agent()
    algo = DQN(agent, memory_size=replay_buffer_len, gamma=discount_factor,
               tau=TAU, batch_size=batch_size, lr=learning_rate,)

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
            action = algo.select_action(env, obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            algo.update(torch.tensor(obs, dtype=torch.float32),
                        torch.tensor([action], dtype=torch.long),
                        torch.tensor([reward], dtype=torch.float32),
                        None if terminated else torch.tensor(next_obs, dtype=torch.float32),
                        )
            obs = next_obs

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

def save_video(ckpt=None):
    num_eval_episodes = 4

    env = gym.make(ENV_NAME[:-5], render_mode="rgb_array")  # replace with your environment
    env = RecordVideo(env, video_folder="vedio", name_prefix=ENV_NAME,
                      episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
    agent = Agent(ckpt_path=ckpt)

    for episode_num in range(num_eval_episodes):
        obs, info = env.reset()

        episode_over = False
        while not episode_over:
            action, _ = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action.item())

            episode_over = terminated or truncated
    env.close()

    print(f'Episode time taken: {env.time_queue}')
    print(f'Episode total rewards: {env.return_queue}')
    print(f'Episode lengths: {env.length_queue}')

def play_video(ckpt=None):
    env = gym.make(ENV_NAME[:-5], render_mode="human")
    agent = Agent(ckpt_path=ckpt)

    observation, info = env.reset()
    episode_over = False
    while not episode_over:
        action, _ = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action.item())

        episode_over = terminated or truncated

    env.close()

def test():
    if False:
    # if True:
        play_video(ckpt=os.path.join('ckpt', 'CartPole-v1_episode_107_reward_478.pth'))
    else:
        save_video(ckpt=os.path.join('ckpt', 'CartPole-v1_episode_107_reward_478.pth'))

if __name__ == '__main__':
    logger.add(os.path.join('train_log', f'{LOG_FILE}'),
               format="{time:HH:mm:ss.SSS} | {file}:{line} | {level} | {message}")
    train()
    # test()
