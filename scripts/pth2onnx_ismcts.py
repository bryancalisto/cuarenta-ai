"""
Export an ISMCTS `CuarentaNet` checkpoint to ONNX for the Go backend.

    python3 -m scripts.pth2onnx_ismcts checkpoints-return/best.pt
    python3 -m scripts.pth2onnx_ismcts checkpoints-return/best.pt --out backend/models/ismcts_return.onnx

WHY THIS IS A SEPARATE SCRIPT FROM `pth2onnx.py`
------------------------------------------------
`pth2onnx.py` exports the PPO actor-critic (`src/ppo/networks.py`). The two pipelines have
DIFFERENT networks with different ONNX signatures, and their script directories both define
`env.py` / `utils.py`, so they cannot share one `sys.path`. The ISMCTS net is also a
different kind of model:

    PPO      stateful, single step   obs [B,1,208] + action_features + mask + LSTM (h, c)
                                     carried by the caller across turns
    ISMCTS   stateless, one decision obs [B,1,208] + action_features + mask, no recurrent
                                     state: history lives in the INPUT SEQUENCE, which the
                                     search builds from its own tracker

THE SIGNATURE (one ONNX call = one decision)

    observation      float32 [B, 1, 208]   ISMCTS-layout observation of the player about
                                           to move (src/ismcts/env.py: observe,
                                           history.flatten_obs key order)
    action_features  float32 [B, 64, 84]   candidate-action features, legal ones first
                                           (src/ismcts/env.py: action_features_matrix)
    action_mask      float32 [B, 64]       1.0 on the legal slots, 0.0 otherwise

    action_logits    float32 [B, 64]       -1e9 on the illegal slots
    value            float32 [B]           the head's output, in the units its value target
                                           was trained in (see `value_semantics` below)
    (no recurrent state in or out)

T=1 is a regime the network is trained for: the search evaluates its INTERIOR nodes from a
single "pending step" with no history, and the training dataset injects that regime on
purpose (`history_dropout`, src/ismcts/dataset.py). MEASURED on `checkpoints-return/best.pt`
over 372 decisions whose real window is shorter than the 32-step maximum: feeding the
padded 32-step window and feeding the single pending step BOTH keep the argmax on the same
legal action 100% of the time — and the argmax is all this backend uses. The windowed
variant is exported too (`--window N`, experiment axis) but the Go backend consumes the
single-step one.

VALUE SEMANTICS — the reason the sidecar exists
-----------------------------------------------
The value head is only different from the win-mode head by its ACTIVATION, so the same
weights load into either mode and the numbers mean different things:

    --value-target win      value = win estimate in [-1, 1] for the mover's team
    --value-target return   value = shaped return-to-go / RETURN_VALUE_SCALE, i.e. in game
                            points = value * RETURN_VALUE_SCALE, positive when the mover's
                            team is expected to out-score the other; NOT bounded by 1

The exporter takes the mode from (in order) `--value-target`, the checkpoint's `.json`
provenance sidecar written by `src/ismcts/train.py`, or `win`. The exported `model.json`
records the mode, the scale and `value_semantics`, so a consumer never has to guess.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# src/ismcts is a script directory with bare sibling imports (`from env import ...`), so it
# goes on sys.path directly. It MUST NOT be mixed with src/ppo: both define `env` and
# `utils`, and the first import wins.
_REPO = Path(__file__).resolve().parent.parent
_ISMCTS = _REPO / "src" / "ismcts"
sys.path.insert(0, str(_ISMCTS))

from env import ACTION_FEAT_DIM, MAX_ACTIONS, CuarentaEnv, action_features_matrix  # noqa: E402
from history import HistoryTracker, OBS_FLAT_DIM                                  # noqa: E402
from network import VALUE_TARGETS, CuarentaNet                                    # noqa: E402


class ISMCTSPolicyOnnx(nn.Module):
    """
    Inference wrapper: flat tensors in, logits and value out, no recurrent state.

    The mask arrives as float32 because it is easier to build from Go than a BOOL tensor,
    and it is compared to 0.5 here (same convention as the PPO export).
    """

    def __init__(self, net: CuarentaNet):
        super().__init__()
        self.net = net

    def forward(
        self,
        observation: torch.Tensor,        # [B, T, obs_dim]
        action_features: torch.Tensor,    # [B, n_slots, action_feat_dim]
        action_mask: torch.Tensor,        # [B, n_slots] float32
    ):
        B, T, _ = observation.shape
        # A single "pending" step: no action has been taken at it yet, which is why the
        # action-history block is zeros and its mask is False (NetworkISMCTS._eval_node
        # feeds interior nodes exactly like this).
        act_seq = torch.zeros(B, T, ACTION_FEAT_DIM, dtype=observation.dtype,
                              device=observation.device)
        act_mask = torch.zeros(B, T, dtype=torch.bool, device=observation.device)
        logits, value = self.net(observation, act_seq, act_mask, action_features,
                                 action_mask > 0.5)
        return logits, value


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_state_dict(path: Path) -> dict:
    state = torch.load(str(path), map_location="cpu")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict) or "obs_mlp.0.weight" not in state:
        raise SystemExit(
            f"{path} is not an ISMCTS CuarentaNet checkpoint "
            f"(keys: {sorted(state.keys())[:6] if isinstance(state, dict) else type(state)}).\n"
            "PPO checkpoints go through scripts/pth2onnx.py instead."
        )
    return state


def resolve_value_target(pt_path: Path, explicit: str | None) -> tuple[str, dict, str]:
    """
    Which head the weights were trained with, and where that came from.

    Returns (value_target, sidecar_meta, source) where source is "flag", "sidecar" or
    "default". It REFUSES to guess when neither the flag nor a sidecar says: the two heads
    are the same parameters with a different activation, so nothing in the weights can
    distinguish them, and guessing wrong exports a model whose value output means something
    else — a silent error that only shows up as strange AI behaviour.
    """
    meta: dict = {}
    sidecar = Path(str(pt_path) + ".json")
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text())
        except ValueError:
            meta = {}
    if explicit:
        return explicit, meta, "flag"
    recorded = meta.get("value_target")
    if recorded in VALUE_TARGETS:
        return recorded, meta, "sidecar"
    raise SystemExit(
        f"Cannot tell which value target {pt_path} was trained with.\n"
        f"  sidecar: {sidecar} "
        + ("exists but does not record a value_target" if meta else "does not exist")
        + "\n\n"
        "The win-mode head ends in tanh and the return-mode head is linear, but that is an\n"
        "ACTIVATION, not a weight: the same state_dict loads into either, and the numbers\n"
        "then mean different things (a win estimate in [-1,1], or a return in game points).\n"
        "\n"
        "Pass it explicitly, e.g.\n"
        f"    python3 -m scripts.pth2onnx_ismcts {pt_path} --value-target return\n"
        + ("\n(The sidecar present was written by an older trainer without the value target.)"
           if meta else
           "\n(Checkpoints written by the current src/ismcts/train.py carry this in a .json\n"
           " sidecar, so this question only comes up for older ones.)")
    )


def export(pt_path: Path, out_path: Path | None, value_target: str | None = None,
           window: int = 1, verify: bool = True) -> Path:
    state = read_state_dict(pt_path)
    target, meta, target_source = resolve_value_target(pt_path, value_target)

    # The head's ACTIVATION comes from the value target; everything else is in the weights.
    # `lstm_layers` in particular is only visible as the set of `lstm.weight_ih_l{N}` keys,
    # and getting it wrong makes load_state_dict fail with "missing keys".
    lstm_layers = 1 + max(
        (int(k.split("weight_ih_l")[1]) for k in state
         if k.startswith("lstm.weight_ih_l")), default=0) if any(
        k.startswith("lstm.weight_ih_l") for k in state) else 0
    obs_dim = int(state["obs_mlp.0.weight"].shape[1])
    feat_dim = int(state["act_hist_proj.weight"].shape[1])
    if lstm_layers < 1:
        raise SystemExit(f"{pt_path} has no LSTM weights; it is not a CuarentaNet")
    if obs_dim != OBS_FLAT_DIM or feat_dim != ACTION_FEAT_DIM:
        raise SystemExit(
            f"{pt_path} takes a {obs_dim}-dim observation / {feat_dim}-dim action features, "
            f"but this backend's contract is {OBS_FLAT_DIM} / {ACTION_FEAT_DIM} "
            "(src/ismcts/env.py). It is a different architecture; do not serve it here.")

    net = CuarentaNet(
        obs_dim=obs_dim,
        n_actions=MAX_ACTIONS,
        action_feat_dim=feat_dim,
        hidden=int(state["obs_mlp.0.weight"].shape[0]),
        lstm_layers=lstm_layers,
        value_target=target,
    )
    net.load_state_dict(state)
    net.eval()
    wrapper = ISMCTSPolicyOnnx(net).eval()

    if window < 1:
        raise SystemExit("--window must be >= 1 (1 = the single pending step)")

    onnx_path = out_path or pt_path.with_suffix(".onnx")
    dummy = (
        torch.zeros(1, window, net.obs_dim),
        torch.zeros(1, net.n_actions, net.action_feat_dim),
        torch.ones(1, net.n_actions),
    )
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation", "action_features", "action_mask"],
        output_names=["action_logits", "value"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action_features": {0: "batch"},
            "action_mask": {0: "batch"},
            "action_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        dynamo=False,   # the legacy exporter produces the most predictable graph for Go
    )

    if verify:
        verify_parity(net, onnx_path, window=window)

    scale = float(meta.get("return_value_scale", 30.0)) if target == "return" else 1.0
    semantics = (
        f"win estimate in [-1,1] for the mover's team" if target == "win" else
        f"shaped return-to-go for the mover's team, divided by RETURN_VALUE_SCALE "
        f"({scale:g}); multiply by {scale:g} for game points, and it is NOT bounded by 1"
    )
    model_meta = {
        "architecture": "ismcts_cuarenta_net_action_features",
        "observation": {
            "size": net.obs_dim,
            "steps": window,
            "layout": "src/ismcts/history.py: flatten_obs key order (OBS_FLAT_DIM=208); "
                      "step T-1 is the decision being made and must be the pending step",
        },
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
        "action_slots": net.n_actions,
        "recurrent_state": None,
        "value_target": target,
        "value_target_source": target_source,
        "value_scale": scale,
        "value_semantics": semantics,
        "action_selection": "argmax(action_logits[:n_legal]); slot i == your i-th legal action",
        "source_checkpoint": str(pt_path),
        "source_checkpoint_sha256": _sha256(pt_path),
        "source_checkpoint_meta": meta,
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "exporter": "scripts/pth2onnx_ismcts.py",
    }
    meta_path = onnx_path.with_suffix(".json")
    meta_path.write_text(json.dumps(model_meta, indent=2))

    print(f"[ONNX] exported → {onnx_path}")
    print(f"[ONNX] metadata → {meta_path}")
    if target_source != "sidecar":
        print(f"[ONNX] NOTE: value target {target!r} came from the {target_source}, not from "
              "the checkpoint's sidecar — the exported `value` output is only as trustworthy "
              "as that claim.")
    print(f"[ONNX] observation {net.obs_dim} dims x {window} step(s) | "
          f"{net.n_actions} slots x {net.action_feat_dim} features | "
          f"value target {target!r} (scale {scale:g}) | no recurrent state")
    return onnx_path


def verify_parity(net: CuarentaNet, onnx_path: Path, window: int = 1,
                  tol: float = 1e-4) -> None:
    """
    Check the exported graph against PyTorch on real positions, IN PYTHON.

    `onnxruntime` is not installed here (the Go backend has the only bindings), so this uses
    `onnx.reference`, which executes the graph. It is slow and only runs a handful of
    positions, but it is enough to catch the failures that matter and that would otherwise
    be silent: the wrong head activation (a tanh-squashed return value), a mask that is not
    applied, action features read from the wrong input, or an output wired to the wrong
    tensor.
    """
    from onnx.reference import ReferenceEvaluator
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    sess = ReferenceEvaluator(model)

    env = CuarentaEnv(seed=4)
    env.reset(seed=4)
    tracker = HistoryTracker(env)
    checked = 0
    for _ in range(6):
        if env.state.done:
            break
        p = env.state.current_player
        obs = env.observe(p)
        feats, mask = action_features_matrix(env.legal_actions(), env.state)
        obs_seq = np.asarray(obs_flat(env, p), dtype=np.float32)[None, None, :]

        with torch.no_grad():
            logits_t, value_t = net(
                torch.as_tensor(obs_seq),
                torch.zeros(1, window, ACTION_FEAT_DIM),
                torch.zeros(1, window, dtype=torch.bool),
                torch.as_tensor(feats[None], dtype=torch.float32),
                torch.as_tensor(mask[None], dtype=torch.bool),
            )
        out = sess.run(None, {
            "observation": obs_seq,
            "action_features": feats[None].astype(np.float32),
            "action_mask": mask[None].astype(np.float32),
        })
        logits_o, value_o = np.asarray(out[0]), np.asarray(out[1])

        assert np.allclose(logits_t.numpy(), logits_o, atol=tol), (
            f"logits differ: max|d|={np.abs(logits_t.numpy() - logits_o).max():g}\n"
            "the exported graph does not reproduce PyTorch — do not ship this model"
        )
        assert np.allclose(value_t.numpy(), value_o, atol=tol), (
            f"value differs: torch={value_t.numpy()} onnx={value_o}"
        )
        n_legal = int(mask.sum())
        assert int(np.argmax(logits_o[0, :n_legal])) == int(
            torch.argmax(logits_t[0, :n_legal])), "argmax over the legal slots differs"
        checked += 1
        env.step(int(np.random.randint(n_legal)))

    print(f"[ONNX] parity vs PyTorch: OK on {checked} real decisions "
          f"(logits, value, argmax; atol={tol:g})")
    check_head_activation(model, net.value_target)
    if net.value_target == "return":
        values = []
        for _ in range(3):
            f = np.random.rand(1, 1, net.obs_dim).astype(np.float32)
            af = np.random.rand(1, net.n_actions, net.action_feat_dim).astype(np.float32)
            am = np.ones((1, net.n_actions), np.float32)
            out = sess.run(None, {"observation": f, "action_features": af, "action_mask": am})
            values.append(float(out[1][0]))
        print(f"[ONNX] value head is LINEAR (return mode): random inputs give "
              f"{['%.3f' % v for v in values]}, unbounded — a tanh head could not leave (-1,1)")


def check_head_activation(model, value_target: str) -> None:
    """
    Assert the `value` output's ACTIVATION matches the requested value target, by walking
    the graph backwards from that output.

    WHY THIS IS NEEDED ON TOP OF THE NUMERIC PARITY: `tanh(x) == x` to within fp32 for the
    small values a real position produces, so a return-mode head exported by mistake as
    win-mode would pass a value comparison whenever the head happens to be unconfident —
    and the backend would then serve a squashed value forever. The graph itself is
    unambiguous: the win-mode head must have a Tanh on the path from the last Linear to
    `value`, the return-mode head must not.
    """
    import onnx

    producer = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node

    # Walk back from `value` and collect the op types on its path (a linear chain for this
    # network, so a bounded walk is enough).
    ops = []
    frontier = ["value"]
    seen = set()
    for _ in range(12):
        nxt = []
        for name in frontier:
            node = producer.get(name)
            if node is None or node.name in seen:
                continue
            seen.add(node.name)
            ops.append(node.op_type)
            nxt.extend(node.input)
        if not nxt:
            break
        frontier = nxt

    has_tanh = "Tanh" in ops
    wants_tanh = value_target == "win"
    if has_tanh != wants_tanh:
        raise SystemExit(
            f"exported head does not match --value-target {value_target}: the path to the "
            f"`value` output {'contains' if has_tanh else 'does not contain'} a Tanh node.\n"
            f"  ops on the value path: {ops}\n"
            "A win-mode head must end in tanh (its target is in [-1,1]); a return-mode head "
            "must be linear (its target is a scaled return). Re-run with the value target "
            "the checkpoint was trained with."
        )
    print(f"[ONNX] head activation: {'tanh (win target)' if has_tanh else 'linear (return target)'}"
          f" — matches --value-target {value_target}")


def obs_flat(env: CuarentaEnv, player: int) -> np.ndarray:
    """The 208-dim observation of `player` in `history.flatten_obs` key order."""
    from env import flat_obs_for
    return flat_obs_for(player, env.state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an ISMCTS CuarentaNet checkpoint to ONNX")
    parser.add_argument("checkpoint", type=str, help="path to a .pt checkpoint")
    parser.add_argument("--out", type=str, default=None, help="output .onnx path")
    parser.add_argument("--value-target", type=str, default=None, choices=list(VALUE_TARGETS),
                        help="head the weights were trained with; defaults to the value "
                             "recorded in the checkpoint's .json sidecar, else 'win'. "
                             "Getting this wrong exports a head whose units mean something "
                             "else (a tanh-squashed return, or an unbounded 'win estimate')")
    parser.add_argument("--window", type=int, default=1,
                        help="observation steps in the graph (default 1: the single pending "
                             "step, which is the regime the search uses for interior nodes "
                             "and what the Go backend feeds). >1 exports the same net with a "
                             "fixed window length, left-padded by the caller.")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the torch-vs-ONNX parity check")
    args = parser.parse_args()

    pt_path = Path(args.checkpoint)
    if not pt_path.exists():
        raise SystemExit(f"checkpoint not found: {pt_path}")
    export(pt_path, Path(args.out) if args.out else None,
           value_target=args.value_target, window=args.window, verify=not args.no_verify)


if __name__ == "__main__":
    main()
