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
from collections import namedtuple, deque
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


ENV_NAME = 'InvertedPendulum-v5'
LOG_FILE = f'{ENV_NAME}.txt'
SEED = 1111

# global parameters
g_num_episodes = 1000
g_discount_factor = 0.99
g_learning_rate = 5e-4
g_stop_reward = 950
g_batch_size = 128
g_replay_buffer_size = 100000  # trick, use large replay buffer size
g_target_update_tau = 0.005

# tricks
g_gradient_clip = 0.  # gradient clip, set 0. to disable
g_noise_std_scale = 0.05  # explore noise std, set 0. to disable


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'terminated'))

class ActorNet(nn.Module):
    def  __init__(self, n_states, n_actions, action_bound_scale, action_bound_bias):
        super().__init__()

        self.action_bound_scale = action_bound_scale
        self.action_bound_bias = action_bound_bias

        self.fc = nn.Sequential(
            nn.Linear(n_states, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, n_actions),
            nn.Tanh(),  # fix action in [-1, 1]
        )

    def forward(self, state):
        action = self.fc(state)
        action = action * self.action_bound_scale + self.action_bound_bias
        return action

class CriticNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_states + n_actions, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        state_value = self.fc(torch.cat((state, action), 1))
        return state_value

class Agent:
    def __init__(self, actor_ckpt=None, critic_ckpt=None, noise_std_scale=0.):
        action_bound_low, action_bound_high = -3, 3
        action_bound_scale = (action_bound_high - action_bound_low) / 2
        action_bound_bias = action_bound_low + action_bound_scale
        self.noise_std_scale = noise_std_scale
        self.sigma = action_bound_scale * noise_std_scale

        self.actor = ActorNet(4, 1, action_bound_scale, action_bound_bias)
        self.Q_critic = CriticNet(4, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = self.actor.to(self.device)
        self.Q_critic = self.Q_critic.to(self.device)
        self.load(actor_ckpt=actor_ckpt, critic_ckpt=critic_ckpt)
        self.train_mode(False)

    def act(self, state):
        assert isinstance(state, np.ndarray) or isinstance(state, torch.Tensor)
        if torch.is_tensor(state):
            state = state.to(self.device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if self.training:
            action = self.actor(state)
            action = action + self.sigma * torch.randn(action.shape).to(self.device)
        else:
            with torch.no_grad():
                action = self.actor(state)

        return action

    def criticize(self, state, action):
        if self.training:
            q_value = self.Q_critic(state, action)
        else:
            with torch.no_grad():
                q_value = self.Q_critic(state, action)

        return q_value

    def save(self, actor_path, critic_path=None):
        torch.save(self.actor.state_dict(), actor_path)
        if critic_path is not None:
            torch.save(self.Q_critic.state_dict(), critic_path)

    def load(self, actor_ckpt=None, critic_ckpt=None, actor_dict=None, critic_dict=None, tau=1.):
        if actor_ckpt is not None:
            self.actor.load_state_dict(torch.load(actor_ckpt))
        if critic_ckpt is not None:
            self.Q_critic.load_state_dict(torch.load(critic_ckpt))

        if actor_dict is not None:
            temp_state_dict = self.actor.state_dict()
            for key in temp_state_dict.keys():
                temp_state_dict[key] = actor_dict[key] * tau + temp_state_dict[key] * (1 - tau)
            self.actor.load_state_dict(temp_state_dict)
        if critic_dict is not None:
            temp_state_dict = self.Q_critic.state_dict()
            for key in temp_state_dict.keys():
                temp_state_dict[key] = critic_dict[key] * tau + temp_state_dict[key] * (1 - tau)
            self.Q_critic.load_state_dict(temp_state_dict)

    def train_mode(self, training=True):
        self.training = training
        if training:
            self.actor.train()
            self.Q_critic.train()
        else:
            self.actor.eval()
            self.Q_critic.eval()

class DDPG:
    def __init__(self, agent: Agent, gamma=0.99, memory_size=10000, tau=0.005,
                 lr=1e-4, batch_size=0, gradient_clip=0.):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.gradient_clip = gradient_clip

        self.agent.train_mode()
        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.agent.Q_critic.parameters(), lr=lr)
        self.critic_loss_f = nn.MSELoss()

        self.target_agent = Agent()
        self.target_agent.train_mode()
        self.refresh_target_agent()

        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.memory_full = False

    def select_action(self, obs, env=None):
        action = self.agent.act(obs)
        return action.squeeze(0)

    def refresh_target_agent(self, tau=1.0):
        self.target_agent.load(actor_dict=self.agent.actor.state_dict(), critic_dict=self.agent.Q_critic.state_dict(), tau=tau)

    def update(self, transition: Transition):
        self.memory.append(transition)
        if len(self.memory) < self.batch_size:
            return
        if not self.memory_full and len(self.memory) >= self.memory_size:
            logger.info('Memory full.')
            self.memory_full = True

        batch = Transition(*zip(*random.sample(self.memory, self.batch_size)))
        state_batch = torch.stack(batch.state).to(self.agent.device)
        action_batch = torch.stack(batch.action).to(self.agent.device)
        reward_batch = torch.stack(batch.reward).to(self.agent.device)
        next_state_batch = torch.stack(batch.next_state).to(self.agent.device)
        terminal_batch = torch.stack(batch.terminated).to(self.agent.device)

        # critic loss
        with torch.no_grad():
            next_actions = self.target_agent.act(next_state_batch)
            next_q_values = self.target_agent.criticize(next_state_batch, next_actions)
        target_q_values = reward_batch + next_q_values * self.gamma * (1. - terminal_batch)
        q_values = self.agent.criticize(state_batch, action_batch)
        critic_loss = self.critic_loss_f(q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.gradient_clip > 0.:
            torch.nn.utils.clip_grad_norm_(self.agent.Q_critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()

        # actor
        # for params in self.agent.Q_critic.parameters():
        #     params.requires_grad = False
        new_actions = self.agent.act(state_batch)
        new_q_values = self.agent.criticize(state_batch, new_actions)
        actor_loss = -torch.mean(new_q_values)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.gradient_clip > 0.:
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        # for params in self.agent.Q_critic.parameters():
        #     params.requires_grad = True

        self.refresh_target_agent(tau=self.tau)

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
    agent = Agent(noise_std_scale=g_noise_std_scale)
    algo = DDPG(agent, gamma=g_discount_factor, memory_size=g_replay_buffer_size, tau=g_target_update_tau,
                lr=g_learning_rate, batch_size=g_batch_size, gradient_clip=g_gradient_clip)

    logger.info(f'Training agent to play {ENV_NAME} by REINFORCE.')
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
            action = algo.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            done = terminated or truncated
            algo.update(
                transition = Transition(
                    state=torch.tensor(obs, dtype=torch.float32),
                    action=action.detach(),
                    reward=torch.tensor([reward], dtype=torch.float32),
                    next_state=torch.tensor(next_obs, dtype=torch.float32),
                    terminated=torch.tensor([terminated], dtype=torch.float32),
                )
            )
            obs = next_obs

        avg_reward = int(np.mean(env.return_queue))
        cumulative_rewards.append(int(env.return_queue[-1]))
        avg_rewards.append(avg_reward)

        if (episode + 1) % 50 == 0:
            logger.info(f"Episode:{episode + 1}, Average Reward:{avg_reward}")

        if avg_reward > g_stop_reward or episode + 1 == g_num_episodes:
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

    env = gym.make(ENV_NAME, render_mode="rgb_array")  # replace with your environment
    env = RecordVideo(env, video_folder="vedio", name_prefix=ENV_NAME,
                      episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
    agent = Agent(actor_ckpt=ckpt)

    for episode_num in range(num_eval_episodes):
        obs, info = env.reset()

        episode_over = False
        while not episode_over:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())

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
        play_video(ckpt=os.path.join('ckpt', 'InvertedPendulum-v5_episode_415_reward_951.pth'))
    else:
        save_video(ckpt=os.path.join('ckpt', 'InvertedPendulum-v5_episode_415_reward_951.pth'))

if __name__ == '__main__':
    logger.add(os.path.join('train_log', f'{LOG_FILE}'),
               format="{time:HH:mm:ss.SSS} | {file}:{line} | {level} | {message}")
    # train()
    test()
