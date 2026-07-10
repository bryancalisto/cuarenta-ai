from typing import Optional, Tuple

from torch import nn
import torch


class PolicyNetwork(nn.Module):
    """Neural network policy for action selection."""

    def __init__(self, obs_size: int = 121, action_size: int = 200, hidden_size: int = 256):
        """
        obs_size: total observation vector size
        action_size: max number of possible action encodings
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass. obs shape: (batch, obs_size). Output: (batch, action_size)."""
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


LSTMState = Tuple[torch.Tensor, torch.Tensor]


class FlatActorCritic(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        use_lstm: bool = False,
        lstm_hidden: int = 128,
        trunk_hidden: int = 256,
        value_hidden: int = 128,
    ):
        super().__init__()
        self.policy = PolicyNetwork(obs_size=obs_size, action_size=action_size, hidden_size=trunk_hidden)
        self.use_lstm = use_lstm
        self.lstm_hidden = lstm_hidden
        self.trunk_hidden = trunk_hidden

        # LSTM consumes the trunk's output, so its input width tracks trunk_hidden.
        if use_lstm:
            self.lstm = nn.LSTM(input_size=trunk_hidden, hidden_size=lstm_hidden, batch_first=True)
            self.fc3 = nn.Linear(lstm_hidden, action_size)
            feature_size = lstm_hidden
        else:
            feature_size = trunk_hidden

        self.value_head = nn.Sequential(
            nn.Linear(feature_size, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, obs: torch.Tensor, lstm_state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[LSTMState]]:
        x = self.policy.relu(self.policy.fc1(obs))
        x = self.policy.relu(self.policy.fc2(x))

        if self.use_lstm:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x, new_state = self.lstm(x, lstm_state)
            if x.dim() == 3:
                x = x[:, -1, :]
            logits = self.fc3(x)
        else:
            new_state = None
            logits = self.policy.fc3(x)

        value = self.value_head(x).squeeze(-1)
        return logits, value, new_state

    def get_value(self, obs: torch.Tensor, lstm_state: Optional[LSTMState] = None) -> torch.Tensor:
        x = self.policy.relu(self.policy.fc1(obs))
        x = self.policy.relu(self.policy.fc2(x))

        if self.use_lstm:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x, _ = self.lstm(x, lstm_state)
            if x.dim() == 3:
                x = x[:, -1, :]

        return self.value_head(x).squeeze(-1)