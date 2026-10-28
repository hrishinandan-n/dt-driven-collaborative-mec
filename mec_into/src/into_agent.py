"""Algorithm 2: DRL-Based Intelligent Task Offloading Algorithm (INTO).

Implements DDPG-based INTO agent with experience replay, target networks,
exploration noise, deterministic policy gradient updates, and soft target updates.
"""

import os
from typing import Tuple
import numpy as np
import torch
import torch.optim as optim

from .models import ActorNetwork, CriticNetwork


class ReplayMemory:
    """Experience replay memory Omega with capacity |Omega| = 3000."""

    def __init__(self, capacity: int = 3000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        """Stores experience tuple theta(t) = (s(t), a(t), r(t), s(t+1))."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.buffer[i] for i in indices])
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
        )

    def __len__(self):
        return len(self.buffer)


class INTOAgent:
    """Algorithm 2: INTO Agent."""

    def __init__(self, state_dim: int, num_tasks: int = 165, num_bs: int = 3, config: dict = None):
        self.state_dim = state_dim
        self.num_tasks = num_tasks
        self.num_bs = num_bs
        self.action_dim = num_tasks * (num_bs + 1) # 165 * 4 = 660

        # Hyperparameters from config (Table II)
        drl_cfg = config["drl"] if config else {}
        self.lr_actor = drl_cfg.get("learning_rate_actor", 0.0001)
        self.lr_critic = drl_cfg.get("learning_rate_critic", 0.0001)
        self.tau_actor = drl_cfg.get("target_update_tau_actor", 0.001) # eps2
        self.tau_critic = drl_cfg.get("target_update_tau_critic", 0.001) # eps1
        self.gamma = drl_cfg.get("discount_factor", 0.99)
        self.batch_size = drl_cfg.get("batch_size", 64)
        self.replay_capacity = drl_cfg.get("replay_memory_size", 3000)
        self.noise_std = drl_cfg.get("exploration_noise_std", 0.1)

        # Device selection
        self.device = torch.device("cpu")

        # Line 1: Initialize actor and critic networks
        actor_hidden = drl_cfg.get("actor_hidden_layers", [3000, 1000])
        critic_hidden = drl_cfg.get("critic_hidden_layers", [1500, 600])

        self.actor = ActorNetwork(state_dim, self.action_dim, actor_hidden).to(self.device)
        self.critic = CriticNetwork(state_dim, self.action_dim, critic_hidden).to(self.device)

        # Line 2: Initialize target networks
        self.target_actor = ActorNetwork(state_dim, self.action_dim, actor_hidden).to(self.device)
        self.target_critic = CriticNetwork(state_dim, self.action_dim, critic_hidden).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Line 3: Initialize replay memory Omega
        self.memory = ReplayMemory(capacity=self.replay_capacity)

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Line 5: Generate proto-action a_hat(t) = phi(s(t) | theta^phi) + eta(t)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            proto_action = self.actor(state_t).squeeze(0).cpu().numpy()

        if add_noise:
            # Add normally distributed random noise eta(t)
            noise = np.random.normal(0, self.noise_std, size=proto_action.shape)
            proto_action = np.clip(proto_action + noise, 0.0, 1.0)

        # Reshape to (num_tasks, num_bs + 1) -> (165, 4)
        return proto_action.reshape(self.num_tasks, self.num_bs + 1)

    def update(self) -> Tuple[float, float]:
        """Lines 9-13: Sample mini-batch, update Critic, Actor, and Target networks."""
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0

        # Line 9: Randomly sample mini-batch Lambda
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # Line 10: Compute target value for critic network: rho = r + gamma * Q'(s', phi'(s'))
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, target_actions)
            target_value = rewards + self.gamma * target_q

        # Line 11: Update critic parameters by minimizing MSE loss: L(theta^Q) = (rho - Q(s, a))^2
        self.critic.train()
        current_q = self.critic(states, actions)
        critic_loss = torch.mean((target_value - current_q) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Line 12: Update actor parameters with deterministic policy gradient (Eq. 20)
        # J = E[Q(s, phi(s))] -> maximize Q
        self.actor.train()
        actor_actions = self.actor(states)
        actor_loss = -torch.mean(self.critic(states, actor_actions))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Line 13: Update target networks according to Eq. (21) and (22)
        # theta^Q' = eps1 * theta^Q + (1 - eps1) * theta^Q'
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau_critic * param.data + (1.0 - self.tau_critic) * target_param.data)

        # theta^phi' = eps2 * theta^phi + (1 - eps2) * theta^phi'
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau_actor * param.data + (1.0 - self.tau_actor) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_actor.load_state_dict(checkpoint["target_actor_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
