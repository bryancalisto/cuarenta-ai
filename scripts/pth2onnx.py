import sys
from pathlib import Path
import torch
import torch.onnx
import torch.nn as nn

from src.networks import PolicyNetwork


class PolicyNetworkOnly(nn.Module):
    """Wrapper that only outputs action logits (for ONNX inference)."""

    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.policy = PolicyNetwork(obs_size=obs_size, action_size=action_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy(obs)


class FlatActorCriticPolicyOnly(nn.Module):
    """Actor-critic wrapper that only outputs action logits."""

    def __init__(self, obs_size: int, action_size: int, use_lstm: bool = False, lstm_hidden: int = 128):
        super().__init__()
        self.policy = PolicyNetwork(obs_size=obs_size, action_size=action_size)
        self.use_lstm = use_lstm

        if use_lstm:
            self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden, batch_first=True)
            self.fc3 = nn.Linear(lstm_hidden, action_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.policy.relu(self.policy.fc1(obs))
        hidden = self.policy.relu(self.policy.fc2(hidden))

        if self.use_lstm:
            hidden = hidden.unsqueeze(1)
            hidden, _ = self.lstm(hidden, None)
            hidden = hidden[:, -1, :]
            return self.fc3(hidden)
        else:
            return self.policy.fc3(hidden)


def save_onnx(pt_path: Path) -> Path:
    """Convert cuarenta policy checkpoint to ONNX format.

    Exports only the action logits output (for use in Go inference engine).
    Handles both LSTM and non-LSTM models.
    """
    pt_filename = pt_path.name
    onnx_filename = pt_filename.replace('.pt', '.onnx')
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

    # Detect checkpoint format
    if 'policy_state' in checkpoint and 'value_state' in checkpoint:
        # PPO FlatActorCritic checkpoint (may include LSTM)
        obs_size = checkpoint.get('obs_size', 121)
        action_size = checkpoint.get('action_size', 200)
        use_lstm = checkpoint.get('use_lstm', False)
        lstm_hidden = 128

        model = FlatActorCriticPolicyOnly(obs_size, action_size, use_lstm=use_lstm, lstm_hidden=lstm_hidden)
        model.policy.load_state_dict(checkpoint['policy_state'])

        if use_lstm and 'lstm_state' in checkpoint and checkpoint['lstm_state'] is not None:
            model.lstm.load_state_dict(checkpoint['lstm_state'])

        model.eval()

        dummy_input = torch.randn(1, obs_size)

        output_path = pt_path.parent / onnx_filename
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
    elif isinstance(checkpoint, dict) and any('fc' in k for k in checkpoint.keys()):
        # Raw PolicyNetwork checkpoint (imitation learning)
        model = PolicyNetworkOnly(obs_size=121, action_size=200)
        model.policy.load_state_dict(checkpoint)
        model.eval()

        dummy_input = torch.randn(1, 121)
        output_path = pt_path.parent / onnx_filename
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
        print(f"[ONNX] Exported policy network → {output_path}")
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