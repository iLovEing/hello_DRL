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
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from collections import deque


ENV_NAME = 'CartPole-v1_offpolicy'
LOG_FILE = f'{ENV_NAME}.txt'
SEED = 1111

num_episodes = 1000
discount_factor = 0.99
learning_rate = 5e-4
stop_reward = 475
batch_size = 64
memory_size = 10000


class ActorNet(nn.Module):
    def __init__(self, n_states, n_actions):
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

    def load(self, actor_ckpt=None, critic_ckpt=None):
        if actor_ckpt is not None:
            self.actor.load_state_dict(torch.load(actor_ckpt))
        if critic_ckpt is not None:
            self.critic.load_state_dict(torch.load(critic_ckpt))

    def train_mode(self, training=True):
        self.training = training
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()


class A2C:
    def __init__(self, agent: Agent, memory_size=10000, gamma=0.99, batch_size=128, lr=1e-4):
        self.agent = agent
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=memory_size)
        self.batch_size = batch_size


        self.agent.train_mode()
        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.agent.critic.parameters(), lr=lr)

        self.critic_loss_f = nn.MSELoss()
        self.critic_step = 0

    def select_action(self, obs, env=None):
        action, prob = self.agent.act(obs)
        return action, prob

    def update(self, obs, action, prob, reward, next_obs, done):
        self.replay_buffer.append({
            'obs': obs,
            'action': action,
            'prob': prob.detach(),
            'reward': reward,
            'next_obs': next_obs
        })
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.stack([x['obs'] for x in batch]).to(self.agent.device)
        action_batch = torch.stack([x['action'] for x in batch]).to(self.agent.device)
        policy_prob_batch = torch.stack([x['prob'] for x in batch]).to(self.agent.device)
        reward_batch = torch.stack([x['reward'] for x in batch]).to(self.agent.device)
        next_states = [x['next_obs'] for x in batch]

        target_prob_batch = self.agent.actor(state_batch).gather(1, action_batch)
        reject_sampleing_factor = target_prob_batch.detach() / policy_prob_batch

        state_values = self.agent.critic(state_batch)
        non_final_mask = torch.tensor(list(map(lambda x: x is not None, next_states)), dtype=torch.bool)
        non_final_next_states = torch.stack(list(filter(lambda x: x is not None, next_states))).to(self.agent.device)
        next_state_values = torch.zeros(state_values.shape, device=self.agent.device)
        next_state_values[non_final_mask, :] = self.agent.critic(non_final_next_states).detach()
        target_state_values = next_state_values * self.gamma + reward_batch

        _factor = torch.sqrt(reject_sampleing_factor)
        # critic_loss = self.critic_loss_f(_factor * state_values, _factor * target_state_values)  # unstable
        critic_loss = self.critic_loss_f(state_values, target_state_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.agent.critic.parameters(), 100)
        self.critic_optimizer.step()

        self.critic_step += 1
        if self.critic_step % 5 != 0:
            return

        advantage = (target_state_values - state_values).detach()
        self.actor_optimizer.zero_grad()
        actor_loss = -torch.mean(reject_sampleing_factor * torch.log(target_prob_batch) * advantage)
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), 100)
        self.actor_optimizer.step()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train():
    set_seed(SEED)

    env = gym.make(ENV_NAME[:-10])
    env = gym.wrappers.RecordEpisodeStatistics(env, 50)
    agent = Agent()
    algo = A2C(agent, memory_size=memory_size, gamma=discount_factor, batch_size=batch_size, lr=learning_rate)

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
            action, prob = algo.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            algo.update(torch.tensor(obs, dtype=torch.float32),
                        action,
                        prob,
                        torch.tensor([reward], dtype=torch.float32),
                        None if terminated else torch.tensor(next_obs, dtype=torch.float32),
                        done)
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

    env = gym.make(ENV_NAME[:-10], render_mode="rgb_array")  # replace with your environment
    env = RecordVideo(env, video_folder="vedio", name_prefix=ENV_NAME,
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
    env = gym.make(ENV_NAME[:-10], render_mode="human")
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
        play_video(ckpt=os.path.join('ckpt', 'CartPole-v1_offpolicy_episode_135_reward_478.pth'))
    else:
        save_video(ckpt=os.path.join('ckpt', 'CartPole-v1_offpolicy_episode_135_reward_478.pth'))

if __name__ == '__main__':
    logger.add(os.path.join('train_log', f'{LOG_FILE}'),
               format="{time:HH:mm:ss.SSS} | {file}:{line} | {level} | {message}")
    # train()
    test()
