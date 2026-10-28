"""Double Deep Q-Network (DDQN) Baseline for Task Offloading.

As specified in Section V-B: Contains two fully connected layers (1000 and 600 neurons).
Learning rate = 0.001, target update rate = 0.001. Evaluates value-based DRL.
"""

import os
from typing import Tuple
import numpy as np
import torch
import torch.optim as optim

from ..models import DDQNNetwork


class DDQNReplayMemory:
    """Replay memory for DDQN transitions."""

    def __init__(self, capacity: int = 3000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action_scores: np.ndarray, reward: float, next_state: np.ndarray):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            torch.FloatTensor(state),
            torch.FloatTensor(action_scores),
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


class DDQNBaseline:
    """Double DQN Baseline Agent."""

    def __init__(self, state_dim: int, num_tasks: int = 165, num_bs: int = 3, config: dict = None):
        self.state_dim = state_dim
        self.num_tasks = num_tasks
        self.num_bs = num_bs
        self.action_dim = num_tasks * (num_bs + 1) # 660 action values

        drl_cfg = config["drl"] if config else {}
        self.lr = drl_cfg.get("learning_rate_ddqn", 0.001)
        self.tau = drl_cfg.get("target_update_tau_ddqn", 0.001)
        self.gamma = drl_cfg.get("discount_factor", 0.99)
        self.batch_size = drl_cfg.get("batch_size", 64)
        self.replay_capacity = drl_cfg.get("replay_memory_size", 3000)

        self.device = torch.device("cpu")
        hidden_layers = drl_cfg.get("ddqn_hidden_layers", [1000, 600])

        # Online network and Target network
        self.policy_net = DDQNNetwork(state_dim, self.action_dim, hidden_layers).to(self.device)
        self.target_net = DDQNNetwork(state_dim, self.action_dim, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = DDQNReplayMemory(capacity=self.replay_capacity)
        self.epsilon = 0.2

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Selects action values using epsilon-greedy over continuous scores."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        if not evaluate and np.random.rand() < self.epsilon:
            # Exploration
            noise = np.random.uniform(-0.5, 0.5, size=q_values.shape)
            action_scores = q_values + noise
        else:
            action_scores = q_values

        return action_scores.reshape(self.num_tasks, self.num_bs + 1)

    def update(self) -> float:
        """Double DQN parameter update: Q_target = r + gamma * Q_target(s', argmax Q_online(s'))."""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # DDQN target calculation
        with torch.no_grad():
            next_online_q = self.policy_net(next_states)
            next_target_q = self.target_net(next_states)
            # Reshape to (batch, tasks, 4)
            b = states.size(0)
            next_online_reshaped = next_online_q.view(b, self.num_tasks, self.num_bs + 1)
            next_target_reshaped = next_target_q.view(b, self.num_tasks, self.num_bs + 1)

            # Argmax over online network
            best_actions = torch.argmax(next_online_reshaped, dim=-1, keepdim=True)
            # Evaluate using target network
            best_q = torch.gather(next_target_reshaped, dim=-1, index=best_actions).sum(dim=1)
            target = rewards + self.gamma * best_q

        self.policy_net.train()
        current_q_all = self.policy_net(states)
        current_q_reshaped = current_q_all.view(b, self.num_tasks, self.num_bs + 1)
        # Sum chosen action values
        action_indices = torch.argmax(actions.view(b, self.num_tasks, self.num_bs + 1), dim=-1, keepdim=True)
        current_q = torch.gather(current_q_reshaped, dim=-1, index=action_indices).sum(dim=1)

        loss = torch.mean((target - current_q) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network soft update
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
