"""
Export a Cuarenta policy checkpoint to ONNX for the Go backend.

Usage (from the repository root):

    python3 -m scripts.pth2onnx checkpoints-ppo/best.pt
    python3 -m scripts.pth2onnx checkpoints-ppo/bootstrap.pt --out /tmp/model.onnx

WHAT IS EXPORTED
----------------
The PPO network is an actor-critic with an ACTION-FEATURE policy head (the same
representation the ISMCTS pipeline uses). One ONNX call is one decision:

    observation      float32 [B, 1, 208]   ISMCTS-layout observation of the player about
                                           to move (src/ppo/utils.py: OBS_KEYS_ISMCTS)
    action_features  float32 [B, 64, 84]   candidate-action features of that decision
                                           (src/ppo/env.py: action_features_matrix)
    action_mask      float32 [B, 64]       1.0 for the legal slots (packed first)
    lstm_h           float32 [L, B, H]     recurrent state in   (L = layers, H = hidden)
    lstm_c           float32 [L, B, H]     recurrent state in

    action_logits    float32 [B, 64]       -1e9 on illegal slots
    value            float32 [B]           outcome estimate in [-1,1] for the mover's team
    new_lstm_h       float32 [L, B, H]     recurrent state out (feed it back next turn)
    new_lstm_c       float32 [L, B, H]

The caller plays `argmax(action_logits[:n_legal])`, where slot i is the i-th entry of ITS
OWN legal-action list. Build the feature matrix in the same order as the list you index
with the result and there is no cross-language ordering contract to maintain — that is
the point of a feature-based head.

A `model.json` describing all of this is written next to the .onnx, so a consumer can
assert that it is feeding the model what it expects.

Legacy flat-action-space checkpoints (244-dim observation, 1640 logits) are rejected with
an explanation: they are a different architecture and cannot be exported here.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# src/ppo is written as a script directory (bare sibling imports), so put it on
# sys.path rather than importing it as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ppo"))
from env import ACTION_FEAT_DIM, MAX_ACTIONS           # noqa: E402
from networks import ActorCriticNetwork, load_checkpoint_file  # noqa: E402
from utils import OBS_KEYS_ISMCTS, OBS_SIZE_ISMCTS     # noqa: E402


class PPOPolicyOnnx(nn.Module):
    """
    Inference wrapper giving the ONNX graph a flat, explicit signature.

    The mask arrives as float32 (easier to build from Go than a BOOL tensor) and is
    converted here; the LSTM state is passed through so the caller can carry it across
    turns of a game.
    """

    def __init__(self, net: ActorCriticNetwork):
        super().__init__()
        self.net = net

    def forward(
        self,
        observation: torch.Tensor,        # [B, 1, obs_dim]
        action_features: torch.Tensor,    # [B, n_slots, action_feat_dim]
        action_mask: torch.Tensor,        # [B, n_slots] float32
        lstm_h: torch.Tensor,             # [L, B, hidden]
        lstm_c: torch.Tensor,             # [L, B, hidden]
    ):
        logits, value, new_state = self.net(
            observation, action_features, action_mask > 0.5, (lstm_h, lstm_c)
        )
        new_h, new_c = new_state if new_state is not None else (lstm_h, lstm_c)
        return logits, value, new_h, new_c


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Hash a checkpoint so an export can be tied back to the exact weights it came from."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def export(pt_path: Path, out_path: Path | None = None) -> Path:
    state = load_checkpoint_file(str(pt_path), torch.device("cpu"))
    if "model_state" not in state or "action_feat_dim" not in state:
        raise SystemExit(
            f"{pt_path} is not a current PPO checkpoint (keys: {sorted(state.keys())[:8]}).\n"
            "Legacy flat-action-space models (244-dim observation, 1640 logits) are a\n"
            "different architecture and cannot be exported by this script."
        )

    net = ActorCriticNetwork(
        obs_dim=int(state.get("obs_dim", OBS_SIZE_ISMCTS)),
        action_feat_dim=int(state.get("action_feat_dim", ACTION_FEAT_DIM)),
        n_slots=int(state.get("n_slots", MAX_ACTIONS)),
        hidden=int(state.get("hidden", 256)),
        use_lstm=bool(state.get("use_lstm", True)),
        lstm_hidden=int(state.get("lstm_hidden", 128)),
        lstm_layers=int(state.get("lstm_layers", 1)),
        value_hidden=int(state.get("value_hidden", 128)),
    )
    net.load_state_dict(state["model_state"])
    net.eval()
    wrapper = PPOPolicyOnnx(net).eval()

    layers = net.lstm_layers if net.use_lstm else 1
    hidden = net.lstm_hidden if net.use_lstm else 1

    onnx_path = out_path or pt_path.with_suffix(".onnx")
    dummy = (
        torch.zeros(1, 1, net.obs_dim),
        torch.zeros(1, net.n_slots, net.action_feat_dim),
        torch.ones(1, net.n_slots),
        torch.zeros(layers, 1, hidden),
        torch.zeros(layers, 1, hidden),
    )

    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation", "action_features", "action_mask", "lstm_h", "lstm_c"],
        output_names=["action_logits", "value", "new_lstm_h", "new_lstm_c"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action_features": {0: "batch"},
            "action_mask": {0: "batch"},
            "lstm_h": {1: "batch"},
            "lstm_c": {1: "batch"},
            "action_logits": {0: "batch"},
            "value": {0: "batch"},
            "new_lstm_h": {1: "batch"},
            "new_lstm_c": {1: "batch"},
        },
        dynamo=False,   # the legacy exporter produces the most predictable graph for Go
    )

    meta = {
        "architecture": "ppo_actor_critic_action_features",
        "observation": {"keys": list(OBS_KEYS_ISMCTS), "size": net.obs_dim},
        "action_features": {
            "size": net.action_feat_dim,
            "layout": {
                "[0:40]": "played card one-hot",
                "[40:80]": "captured cards one-hot",
                "[80]": "is_capture",
                "[81]": "is_caida",
                "[82]": "is_limpia",
                "[83]": "unused",
            },
        },
        "action_slots": net.n_slots,
        "use_lstm": net.use_lstm,
        "lstm_layers": net.lstm_layers if net.use_lstm else 0,
        "lstm_hidden": net.lstm_hidden if net.use_lstm else 0,
        "value_semantics": "outcome estimate in [-1,1] for the mover's team",
        "action_selection": "argmax(action_logits[:n_legal]); slot i == your i-th legal action",
        "source_checkpoint": str(pt_path),
        "source_checkpoint_sha256": _sha256(pt_path),
        "source_step": int(state.get("step", 0)),
        # Stamped so a stale export is identifiable: the Go parity test compares this
        # hash with the checkpoint the fixture was generated from, and refuses to
        # pretend an old fixture still validates a freshly exported model.
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "exporter": "scripts/pth2onnx.py",
    }
    meta_path = onnx_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[ONNX] exported → {onnx_path}")
    print(f"[ONNX] metadata → {meta_path}")
    print(f"[ONNX] observation {net.obs_dim} dims | {net.n_slots} slots x "
          f"{net.action_feat_dim} features | lstm={'yes' if net.use_lstm else 'no'}"
          + (f" (layers={net.lstm_layers}, hidden={net.lstm_hidden})" if net.use_lstm else ""))
    return onnx_path


def verify(onnx_path: Path) -> None:
    """Structural check of the exported graph (requires the `onnx` package)."""
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("[ONNX] checker: OK")
    for kind, values in (("in ", model.graph.input), ("out", model.graph.output)):
        for value in values:
            dims = [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]
            print(f"[ONNX]   {kind} {value.name:16} {dims}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Cuarenta PPO checkpoint to ONNX")
    parser.add_argument("checkpoint", type=str, help="path to a .pt checkpoint")
    parser.add_argument("--out", type=str, default=None, help="output .onnx path")
    parser.add_argument("--no-verify", action="store_true", help="skip the graph check")
    args = parser.parse_args()

    pt_path = Path(args.checkpoint)
    if not pt_path.exists():
        raise SystemExit(f"checkpoint not found: {pt_path}")

    onnx_path = export(pt_path, Path(args.out) if args.out else None)
    if not args.no_verify:
        try:
            verify(onnx_path)
        except ImportError:
            print("[ONNX] (install `onnx` to run the graph check)")


if __name__ == "__main__":
    main()
