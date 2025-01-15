import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"Replay buffer saved to {filepath}")

    def load_buffer(self, filepath):
        with open(filepath, 'rb') as f:
            self.buffer = pickle.load(f)
        self.position = len(self.buffer) % self.buffer_size
        print(f"Replay buffer loaded from {filepath}")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.out(x))
        return x * self.action_bound


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, buffer_size=100000, batch_size=128, gamma=0.99, tau=0.002, actor_lr=0.0001, critic_lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.buffer = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.critic = Critic(state_dim, action_dim)

        # Target networks
        self.target_actor = Actor(state_dim, action_dim, action_bound)
        self.target_critic = Critic(state_dim, action_dim)

        # Initialize target networks with same weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def policy(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).numpy()
        return action

    def noisy_policy(self, state, noise_scale=0.1):
        action = self.policy(state)
        noise = np.random.normal(0, noise_scale, size=self.action_dim)
        return np.clip(action + noise, -self.action_bound, self.action_bound)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample()

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Update Critic
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            noise = torch.clamp(torch.normal(0, 0.2, size=target_actions.shape), -0.5, 0.5)
            target_actions = torch.clamp(target_actions + noise, -self.action_bound, self.action_bound)

            target_q_values = self.target_critic(next_states, target_actions).squeeze(1)
            y = rewards + self.gamma * (1 - dones) * target_q_values

        q_values = self.critic(states, actions).squeeze(1)
        critic_loss = nn.MSELoss()(q_values, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update Actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft update of target networks
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def save_model(self, prefix):
        torch.save(self.actor.state_dict(), f'{prefix}_actor.pth')
        torch.save(self.critic.state_dict(), f'{prefix}_critic.pth')
        self.buffer.save_buffer(f'{prefix}_replay_buffer.pkl')

    def load_model(self, prefix):
        self.actor.load_state_dict(torch.load(f'{prefix}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{prefix}_critic.pth'))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.buffer.load_buffer(f'{prefix}_replay_buffer.pkl')
        print("Loaded models from disk")
