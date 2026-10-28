"""Neural Network Architectures for INTO and Baseline DDQN.

Matches the paper's exact architecture specifications from Section V and Table II:
- Actor Network: input_dim -> 3000 -> 1000 -> action_dim
- Critic Network: (state_dim + action_dim) -> 1500 -> 600 -> 1
- DDQN Network: state_dim -> 1000 -> 600 -> action_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """INTO Deterministic Policy Network phi(s | theta^phi)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = [3000, 1000]):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Continuous action in [0, 1] range representing offloading preferences
        action = torch.sigmoid(self.fc3(x))
        return action


class CriticNetwork(nn.Module):
    """INTO Q-Value Network Q(s, a | theta^Q)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = [1500, 600]):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class DDQNNetwork(nn.Module):
    """DDQN Baseline Network Q(s, a | theta)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = [1000, 600]):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
