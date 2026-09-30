import sys
from pathlib import Path
import torch
import torch.onnx
import torch.nn as nn

from src.networks import PolicyNetwork


class PolicyNetworkOnly(nn.Module):
    """Wrapper that only outputs action logits (for ONNX inference)."""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.policy = PolicyNetwork(obs_size=obs_size, action_size=action_size, hidden_size=hidden_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy(obs)


class ActorCriticPolicyOnly(nn.Module):
    """Actor-critic wrapper that only outputs action logits."""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int, use_lstm: bool = False, lstm_hidden: int = 128):
        super().__init__()
        self.policy = PolicyNetwork(obs_size=obs_size, action_size=action_size, hidden_size=hidden_size)
        self.use_lstm = use_lstm
        self.lstm_hidden = lstm_hidden

        if use_lstm:
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden, batch_first=True)
            self.fc3 = nn.Linear(lstm_hidden, action_size)

    def forward(self, obs: torch.Tensor, h: torch.Tensor = None, c: torch.Tensor = None) -> torch.Tensor:
        hidden = self.policy.relu(self.policy.fc1(obs))
        hidden = self.policy.relu(self.policy.fc2(hidden))

        if self.use_lstm:
            hidden = hidden.unsqueeze(1)
            if h is None or c is None:
                hidden, (new_h, new_c) = self.lstm(hidden, None)
            else:
                hidden, (new_h, new_c) = self.lstm(hidden, (h, c))
            hidden = hidden[:, -1, :]
            logits = self.fc3(hidden)
            return logits, new_h, new_c
        else:
            return self.policy.fc3(hidden), None, None


def save_onnx(pt_path: Path) -> Path:
    """Convert cuarenta policy checkpoint to ONNX format.

    Exports only the action logits output (for use in Go inference engine).
    Handles both LSTM and non-LSTM models.
    """
    pt_filename = pt_path.name
    onnx_filename = pt_filename.replace('.pt', '.onnx')
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

    # Detect checkpoint format
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        # New PPO ActorCriticNetwork checkpoint format
        obs_size = checkpoint.get('obs_size', 121)
        action_size = checkpoint.get('action_size', 200)
        trunk_hidden = checkpoint.get('trunk_hidden', 256)
        use_lstm = checkpoint.get('use_lstm', False)
        lstm_hidden = checkpoint.get('lstm_hidden', 128)

        model = ActorCriticPolicyOnly(obs_size, action_size, hidden_size=trunk_hidden, use_lstm=use_lstm, lstm_hidden=lstm_hidden)
        model.load_state_dict(checkpoint['model_state'], strict=False)
        model.eval()

        dummy_input = torch.randn(1, obs_size)
        output_path = pt_path.parent / onnx_filename

        if use_lstm:
            # Export with LSTM state tensors: h and c tensors
            # Shape: (num_layers, batch, lstm_hidden) = (1, 1, lstm_hidden)
            dummy_h = torch.zeros(1, 1, lstm_hidden)
            dummy_c = torch.zeros(1, 1, lstm_hidden)

            torch.onnx.export(
                model,
                (dummy_input, dummy_h, dummy_c),
                str(output_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['observation', 'lstm_h', 'lstm_c'],
                output_names=['action_logits', 'new_lstm_h', 'new_lstm_c'],
                dynamic_axes={
                    'observation': {0: 'batch_size'},
                    'lstm_h': {0: 'batch_size'},
                    'lstm_c': {0: 'batch_size'},
                    'action_logits': {0: 'batch_size'},
                    'new_lstm_h': {0: 'batch_size'},
                    'new_lstm_c': {0: 'batch_size'},
                }
            )
        else:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['observation'],
                output_names=['action_logits'],
                dynamic_axes={
                    'observation': {0: 'batch_size'},
                    'action_logits': {0: 'batch_size'},
                }
            )

        print(f"[ONNX] Exported {'LSTM ' if use_lstm else ''}policy → {output_path}")
        return output_path
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python3 -m scripts.pth2onnx <path to .pt>')
        exit(1)

    path = sys.argv[1]
    print(f'Converting {path} to ONNX format')
    pt_path = Path(path)
    print(save_onnx(pt_path))