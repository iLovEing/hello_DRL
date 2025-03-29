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
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


ENV_NAME = 'CartPole-v1'
ALGO_NAME = 'PPO'
LOG_FILE = f'{ENV_NAME}_{ALGO_NAME}.txt'
FIG_FILE = f'{ENV_NAME}_{ALGO_NAME}_reward.png'
INFER_CKPT = 'CartPole-v1_PPO_episode_149_reward_478.pth'

SEED = 1111

num_episodes = 1000
PPO_epochs = 10
discount_factor = 0.99
gae_lambda = 0.95
learning_rate = 5e-4
stop_reward = 475
clip_epsilon = 0.2

Transition = namedtuple('Transition',
                        ('state', 'action', 'action_prob', 'reward', 'next_state'))


class ActorNet(nn.Module):
    def  __init__(self, n_states, n_actions):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_states, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, n_actions)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        action_probs = self.softmax(self.fc(state))
        return action_probs

class CriticNet(nn.Module):
    def __init__(self, n_states):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        state_value = self.fc(state)
        return state_value


class Agent:
    def __init__(self, actor_ckpt=None, critic_ckpt=None):
        self.actor = ActorNet(4, 2)
        self.critic = CriticNet(4)
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
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)

        if self.training:
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs.detach())
            action = action_dist.sample()
        else:
            with torch.no_grad():
                probs = self.actor(state)
                action = torch.argmax(probs)
        return action, probs[0, action]

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
    def __init__(self, agent: Agent, gamma=0.99, lambda_=0.95, ppo_epochs=100, clip_eps=0.2,
                 lr=1e-4):
        self.agent = agent
        self.gamma = gamma
        self.lambda_ = lambda_
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps

        self.agent.train_mode()
        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.agent.critic.parameters(), lr=lr)
        self.critic_loss_f = nn.MSELoss()

        self.episode_buffer = []

    def select_action(self, obs, env=None):
        action, prob = self.agent.act(obs)
        return action, prob

    def estimate_gae(self, state_batch, reward_batch, next_state_batch, terminated):
        with torch.no_grad():
            state_values = self.agent.criticize(state_batch)
            next_state_values = torch.zeros_like(state_values, device=state_values.device)
            if terminated:
                next_state_values[:-1, :] = self.agent.criticize(next_state_batch[:-1, ...])
            else:
                next_state_values = self.agent.criticize(next_state_batch)

            gae = (reward_batch + self.gamma * next_state_values - state_values)
            for i in reversed(range(len(gae) - 1)):
                gae[i] += gae[i + 1] * self.gamma * self.lambda_

            target_state_values = gae + state_values
        return gae, target_state_values

    def update(self, transition: Transition, terminated, truncated):
        self.episode_buffer.append(transition)
        if not (truncated or terminated):
            return

        batch = Transition(*zip(*self.episode_buffer))
        state_batch = torch.stack(batch.state).to(self.agent.device)
        action_batch = torch.stack(batch.action).to(self.agent.device)
        old_prob_batch = torch.stack(batch.action_prob).to(self.agent.device)
        reward_batch = torch.stack(batch.reward).to(self.agent.device)
        next_state_batch = torch.stack(batch.next_state).to(self.agent.device)

        for _ in range(self.ppo_epochs):
            gae, target_state_values = self.estimate_gae(state_batch, reward_batch, next_state_batch, terminated)

            # critic
            state_values = self.agent.criticize(state_batch)
            self.critic_optimizer.zero_grad()
            critic_loss = self.critic_loss_f(state_values, target_state_values)
            critic_loss.backward()
            torch.nn.utils.clip_grad_value_(self.agent.critic.parameters(), 100)
            self.critic_optimizer.step()

            # actor
            prob_batch = self.agent.actor(state_batch)
            prob_batch = prob_batch.gather(1, action_batch)
            ratio = prob_batch / old_prob_batch.clamp_(1e-6)
            surr1 = ratio * gae
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * gae
            actor_loss = -torch.min(surr1, surr2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), 100)
            self.actor_optimizer.step()


        self.episode_buffer.clear()


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
    algo = PPO(agent, gamma=discount_factor, lambda_=gae_lambda, ppo_epochs=PPO_epochs, clip_eps=clip_epsilon,
               lr=learning_rate)

    logger.info(f'Training agent to play {ENV_NAME} by {ALGO_NAME}.')
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
            action, prob = algo.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            algo.update(
                Transition(
                    state=torch.tensor(obs, dtype=torch.float32),
                    action=action,
                    action_prob=prob.detach(),
                    reward=torch.tensor([reward], dtype=torch.float32),
                    next_state=torch.tensor(next_obs, dtype=torch.float32),
                ),
                terminated,
                truncated
            )
            obs = next_obs

        avg_reward = int(np.mean(env.return_queue))
        cumulative_rewards.append(int(env.return_queue[-1]))
        avg_rewards.append(avg_reward)

        if (episode + 1) % 50 == 0:
            logger.info(f"Episode:{episode + 1}, Average Reward:{avg_reward}")

        if avg_reward > stop_reward or episode + 1 == num_episodes:
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
            obs, reward, terminated, truncated, info = env.step(action.item())

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
