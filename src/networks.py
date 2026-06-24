from torch import nn
import torch


class PolicyNetwork(nn.Module):
    """Neural network policy for action selection."""

    def __init__(self, obs_size: int = 121, action_size: int = 200):
        """
        obs_size: total observation vector size
        action_size: max number of possible action encodings
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass. obs shape: (batch, obs_size). Output: (batch, action_size)."""
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
