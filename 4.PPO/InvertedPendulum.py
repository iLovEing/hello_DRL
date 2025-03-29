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
from collections import namedtuple
from torch.distributions.normal import Normal
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


ENV_NAME = 'InvertedPendulum-v5'
ALGO_NAME = 'PPO'
LOG_FILE = f'{ENV_NAME}_{ALGO_NAME}.txt'
FIG_FILE = f'{ENV_NAME}_{ALGO_NAME}_reward.png'
INFER_CKPT = 'InvertedPendulum-v5_PPO_episode_239_reward_951.pth'

SEED = 1111

# global parameters
g_num_episodes = 1000
g_PPO_epochs = 16
g_discount_factor = 0.99
g_gae_lambda = 0.95
g_learning_rate = 5e-4
g_stop_reward = 950
g_ppo_clip_epsilon = 0.2

# tricks
g_entropy_coef = 0.01 # trick 1, policy entropy, set 0. to disable
g_batch_size = 128  # trick 2, use batch data instead of total episode data, set 0 to disable
g_gradient_clip = 0.5  # trick 3, gradient clip, set 0. to disable
g_advantage_norm = True  # trick 4,  advantage normalization
g_state_norm = True
# other tricks
# 1. Beta distribution instead of Guassain distribution
# 2. Orthogonal Initialization

Transition = namedtuple('Transition',
                        ('state', 'action', 'action_log_prob', 'reward', 'next_state', 'terminated', 'truncated'))


class ActorNet(nn.Module):
    def  __init__(self, n_states, n_actions, state_batchnorm=False):
        super().__init__()

        self.state_batchnorm = state_batchnorm
        self.bn = nn.BatchNorm1d(n_states)

        self.fc = nn.Sequential(
            nn.Linear(n_states, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, n_actions * 2)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        if self.state_batchnorm:
            state = self.bn(state)
        output = self.fc(state)
        mean, std = torch.split(output, 1, dim=-1)
        std = torch.log(1 + torch.exp(std))
        return mean, std

class CriticNet(nn.Module):
    def __init__(self, n_states, state_batchnorm=False):
        super().__init__()

        self.state_batchnorm = state_batchnorm
        self.bn = nn.BatchNorm1d(n_states)

        self.fc = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        if self.state_batchnorm:
            state = self.bn(state)
        state_value = self.fc(state)
        return state_value

class Agent:
    def __init__(self, actor_ckpt=None, critic_ckpt=None, state_norm=False):
        self.actor = ActorNet(4, 1, state_batchnorm=state_norm)
        self.critic = CriticNet(4, state_batchnorm=state_norm)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        self.load(actor_ckpt=actor_ckpt, critic_ckpt=critic_ckpt)
        self.train_mode(False)

    def act(self, state):
        assert isinstance(state, np.ndarray) or isinstance(state, torch.Tensor)
        if torch.is_tensor(state):
            state = state.to(self.device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if self.training:
            action_mean, action_std = self.actor(state)
            distrib = Normal(action_mean[0], action_std[0])
            action = distrib.sample()
            log_prob = distrib.log_prob(action)
        else:
            with torch.no_grad():
                action_mean, action_std = self.actor(state)
                distrib = Normal(action_mean[0], action_std[0])
                action = action_mean[0]
                action = torch.clamp(action, -3, 3)
                log_prob = distrib.log_prob(action)

        return action, log_prob

    def criticize(self, state):
        assert isinstance(state, np.ndarray) or isinstance(state, torch.Tensor)
        if torch.is_tensor(state):
            state = state.to(self.device)
        else:
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)

        return self.critic(state)

    def save(self, actor_path, critic_path=None):
        torch.save(self.actor.state_dict(), actor_path)
        if critic_path is not None:
            torch.save(self.critic.state_dict(), critic_path)

    def load(self, actor_ckpt=None, critic_ckpt=None, actor_dict=None):
        assert actor_ckpt is None or critic_ckpt is None
        if actor_ckpt is not None:
            self.actor.load_state_dict(torch.load(actor_ckpt))
        if critic_ckpt is not None:
            self.critic.load_state_dict(torch.load(critic_ckpt))

        if actor_dict is not None:
            self.actor.load_state_dict(actor_dict)

    def train_mode(self, training=True):
        self.training = training
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()

class PPO:
    def __init__(self, agent: Agent, gamma=0.99, lambda_=0.95, ppo_epochs=100, ppo_clip_eps=0.2, entropy_coef=0.,
                 lr=1e-4, batch_size=0, gradient_clip=0., adv_norm=False):
        self.agent = agent
        self.gamma = gamma
        self.lambda_ = lambda_
        self.ppo_epochs = ppo_epochs
        self.ppo_clip_eps = ppo_clip_eps
        self.entropy_coef = entropy_coef

        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        self.adv_norm = adv_norm

        self.agent.train_mode()
        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.agent.critic.parameters(), lr=lr)
        self.critic_loss_f = nn.MSELoss()

        self.memory = []

    def select_action(self, obs, env=None):
        action, log_prob = self.agent.act(obs)
        return action, log_prob

    def estimate_gae(self, state_batch, reward_batch, next_state_batch, terminal_batch, truncated_batch):
        with torch.no_grad():
            state_values = self.agent.criticize(state_batch)
            next_state_values = self.agent.criticize(next_state_batch) * (1. - terminal_batch)

            done_batch = (terminal_batch + truncated_batch).clamp(min=0., max=1.)
            gae = (reward_batch + self.gamma * next_state_values - state_values)
            for i in reversed(range(gae.shape[0] - 1)):
                gae[i] += gae[i + 1] * self.gamma * self.lambda_ * (1.0 - done_batch[i][0])

            target_state_values = gae + state_values
            gae = ((gae - gae.mean()) / (gae.std() + 1e-5)) if self.adv_norm else gae
        return gae, target_state_values

    def update(self, transition: Transition, done):
        self.memory.append(transition)
        # make sure episode end is better, it is beneficial to calculate gae (i think)
        if (self.batch_size > 0 and len(self.memory) < self.batch_size) or not done:
            return

        batch = Transition(*zip(*self.memory))
        state_batch = torch.stack(batch.state).to(self.agent.device)
        action_batch = torch.stack(batch.action).to(self.agent.device)
        old_log_prob_batch = torch.stack(batch.action_log_prob).to(self.agent.device)
        reward_batch = torch.stack(batch.reward).to(self.agent.device)
        next_state_batch = torch.stack(batch.next_state).to(self.agent.device)
        terminal_batch = torch.stack(batch.terminated).to(self.agent.device)
        truncated_batch = torch.stack(batch.truncated).to(self.agent.device)

        for _ in range(self.ppo_epochs):
            gae, target_state_values = self.estimate_gae(state_batch, reward_batch, next_state_batch,
                                                         terminal_batch, truncated_batch)

            # sample
            _indices = torch.randperm(len(self.memory))[:(self.batch_size if self.batch_size > 0 else len(self.memory))]
            _state_batch = state_batch[_indices, ...]
            _target_state_values = target_state_values[_indices, ...]
            _action_batch = action_batch[_indices, ...]
            _old_log_prob_batch = old_log_prob_batch[_indices, ...]
            _gae = gae[_indices, ...]

            # critic
            _state_values = self.agent.criticize(_state_batch)
            _critic_loss = self.critic_loss_f(_state_values, _target_state_values)
            self.critic_optimizer.zero_grad()
            _critic_loss.backward()
            if self.gradient_clip > 0.:
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.gradient_clip)
            self.critic_optimizer.step()

            # actor
            _action_mean, _action_std = self.agent.actor(_state_batch)
            _distrib = Normal(_action_mean, _action_std)
            _dist_entropy = _distrib.entropy().sum(1, keepdim=True)
            _action_log_prob = _distrib.log_prob(_action_batch)
            _ratio = torch.exp(_action_log_prob - _old_log_prob_batch)
            _surr1 = _ratio * _gae
            _surr2 = torch.clamp(_ratio, 1 - self.ppo_clip_eps, 1 + self.ppo_clip_eps) * _gae
            _actor_loss = (-torch.min(_surr1, _surr2) - self.entropy_coef * _dist_entropy).mean()
            self.actor_optimizer.zero_grad()
            _actor_loss.backward()
            if self.gradient_clip > 0.:
                torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.gradient_clip)
            self.actor_optimizer.step()


        self.memory.clear()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train():
    # set_seed(SEED)

    env = gym.make(ENV_NAME)
    env = gym.wrappers.RecordEpisodeStatistics(env, 50)
    agent = Agent()
    algo = PPO(agent, gamma=g_discount_factor, lambda_=g_gae_lambda, ppo_epochs=g_PPO_epochs, ppo_clip_eps=g_ppo_clip_epsilon,
               entropy_coef=g_entropy_coef,lr=g_learning_rate, batch_size=g_batch_size, gradient_clip=g_gradient_clip,
               adv_norm=g_advantage_norm)

    logger.info(f'Training agent to play {ENV_NAME} by {ALGO_NAME}.')
    logger.info(f'num_episodes:{g_num_episodes}, '
                f'discount_rate:{g_discount_factor}, '
                f'learning_rate:{g_learning_rate}, '
                f'seed:{SEED}')

    cumulative_rewards = []
    avg_rewards = []
    for episode in tqdm(range(g_num_episodes)):
        obs, info = env.reset()

        done = False
        while not done:
            action, log_prob = algo.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            done = terminated or truncated
            algo.update(
                transition = Transition(
                    state=torch.tensor(obs, dtype=torch.float32),
                    action=action,
                    action_log_prob=log_prob.detach(),
                    reward=torch.tensor([reward], dtype=torch.float32),
                    next_state=torch.tensor(next_obs, dtype=torch.float32),
                    terminated=torch.tensor([terminated], dtype=torch.float32),
                    truncated=torch.tensor([truncated], dtype=torch.float32),
                ),
                done=done,
            )
            obs = next_obs

        avg_reward = int(np.mean(env.return_queue))
        cumulative_rewards.append(int(env.return_queue[-1]))
        avg_rewards.append(avg_reward)

        if (episode + 1) % 50 == 0:
            logger.info(f"Episode:{episode + 1}, Average Reward:{avg_reward}")

        if avg_reward > g_stop_reward or episode + 1 == g_num_episodes:
            logger.info(f'training finished at episode {episode + 1}, average reward: {avg_reward}')
            agent.save(os.path.join('ckpt', f'{ENV_NAME}_{ALGO_NAME}_episode_{episode + 1}_reward_{avg_reward}.pth'))
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
    scatter_fig.savefig(os.path.join('train_log', FIG_FILE), dpi=400)


def save_video(ckpt=None):
    num_eval_episodes = 4

    env = gym.make(ENV_NAME, render_mode="rgb_array")  # replace with your environment
    env = RecordVideo(env, video_folder="vedio", name_prefix=f'{ENV_NAME}_{ALGO_NAME}',
                      episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
    agent = Agent(actor_ckpt=ckpt)

    for episode_num in range(num_eval_episodes):
        obs, info = env.reset()

        episode_over = False
        while not episode_over:
            action, _ = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

            episode_over = terminated or truncated
    env.close()

    print(f'Episode time taken: {env.time_queue}')
    print(f'Episode total rewards: {env.return_queue}')
    print(f'Episode lengths: {env.length_queue}')

def play_video(ckpt=None):
    env = gym.make(ENV_NAME, render_mode="human")
    agent = Agent(actor_ckpt=ckpt)

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
        play_video(ckpt=os.path.join('ckpt', INFER_CKPT))
    else:
        save_video(ckpt=os.path.join('ckpt', INFER_CKPT))

if __name__ == '__main__':
    logger.add(os.path.join('train_log', f'{LOG_FILE}'),
               format="{time:HH:mm:ss.SSS} | {file}:{line} | {level} | {message}")
    # train()
    test()
