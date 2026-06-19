from pathlib import Path
import torch
import torch.onnx
import torch.nn as nn
from imitation_learning.imitation_learning import PolicyNetwork


class PolicyNetworkOnly(nn.Module):
    """Wrapper that only outputs action logits (for ONNX inference)."""

    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.policy = PolicyNetwork(obs_size=obs_size, action_size=action_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy(obs)


class FlatActorCriticPolicyOnly(nn.Module):
    """Actor-critic wrapper that only outputs action logits."""

    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.policy = PolicyNetwork(obs_size=obs_size, action_size=action_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.policy.relu(self.policy.fc1(obs))
        hidden = self.policy.relu(self.policy.fc2(hidden))
        return self.policy.fc3(hidden)


def save_onnx(pt_path: Path | None = None) -> Path:
    """Convert cuarenta policy checkpoint to ONNX format.

    Exports only the action logits output (for use in Go inference engine).
    """
    if pt_path is None:
        pt_path = Path(__file__).parent.parent.parent / "data" / "policies" / "cuarenta_policy_final.pt"

    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

    # Detect checkpoint format
    if 'policy_state' in checkpoint and 'value_state' in checkpoint:
        # PPO FlatActorCritic checkpoint
        obs_size = checkpoint.get('obs_size', 121)
        action_size = checkpoint.get('action_size', 200)

        model = FlatActorCriticPolicyOnly(obs_size, action_size)
        model.policy.load_state_dict(checkpoint['policy_state'])
        model.eval()

        dummy_input = torch.randn(1, obs_size)

        output_path = pt_path.parent / "cuarenta_policy_final.onnx"
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
        return output_path
    elif isinstance(checkpoint, dict) and any('fc' in k for k in checkpoint.keys()):
        # Raw PolicyNetwork checkpoint (imitation learning)
        model = PolicyNetworkOnly(obs_size=121, action_size=200)
        model.policy.load_state_dict(checkpoint)
        model.eval()

        dummy_input = torch.randn(1, 121)
        output_path = pt_path.parent / "cuarenta_policy_final.onnx"
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
        return output_path
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")


print(save_onnx())