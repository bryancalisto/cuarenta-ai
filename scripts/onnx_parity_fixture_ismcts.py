#!/usr/bin/env python3
"""
Write the JSON fixture that pins PyTorch's ISMCTS policy outputs, so the ONNX export of a
`CuarentaNet` can be checked from Go (which has the only onnxruntime on this machine).

This is the ISMCTS twin of `scripts/onnx_parity_fixture.py` (which does the same for the PPO
actor-critic). It exists as a separate script for the same reason the exporters are separate:
the two pipelines have different networks, different ONNX signatures, and `src/ppo` and
`src/ismcts` both define `env`/`utils`, so they cannot share one `sys.path`.

Usage (from the repository root):

    python3 -m scripts.pth2onnx_ismcts checkpoints-return/best.pt \\
        --value-target return --out backend/models/ismcts_return.onnx
    python3 scripts/onnx_parity_fixture_ismcts.py \\
        --checkpoint checkpoints-return/best.pt \\
        --value-target return \\
        --onnx backend/models/ismcts_return.onnx \\
        --out backend/internal/inference/testdata/ismcts_fixture.json

Then, from `backend/`:

    go test ./internal/inference/ -run TestONNXMatchesPyTorchISMCTS -v

Why it is worth the trouble: the backend serves real players through ONNX, not through
PyTorch. If the export changed semantics (a different head activation, a transposed
action-feature matrix, a missing mask, the observation built in another order), the service
would still run and still return a legal slot — just a worse one — and nothing would
complain. This fixture turns that into a test failure, and it also pins the VALUE target
mode: a return-mode head that got exported as win-mode would come back tanh-squashed, which
the value comparison catches immediately.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "ismcts"))

from env import ACTION_FEAT_DIM, MAX_ACTIONS, CuarentaEnv, action_features_matrix  # noqa: E402
from history import HistoryTracker, flatten_obs                                   # noqa: E402
from network import VALUE_TARGETS, CuarentaNet                                    # noqa: E402

# Decisions are sampled across a game so the fixture covers more than one kind of position
# (empty table, a table with cards, and a later deal).
DECISION_SCHEDULE = [0, 5, 13, 27]
GAME_SEED = 4242


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True, help="the .onnx this fixture validates")
    parser.add_argument("--out", required=True, help="fixture JSON to write")
    parser.add_argument("--value-target", default="return", choices=list(VALUE_TARGETS),
                        help="must match the mode the ONNX was exported with")
    parser.add_argument("--seed", type=int, default=GAME_SEED)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    state = torch.load(str(ckpt), map_location="cpu")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break

    net = CuarentaNet(value_target=args.value_target)
    net.load_state_dict(state)
    net.eval()

    env = CuarentaEnv(seed=args.seed)
    env.reset(seed=args.seed)
    tracker = HistoryTracker(env)
    rng = np.random.default_rng(args.seed)

    decisions = []
    step = 0
    while not env.state.done and len(decisions) < len(DECISION_SCHEDULE):
        player = env.state.current_player
        obs = env.observe(player)
        legal = env.legal_actions()
        feats, mask = action_features_matrix(legal, env.state)
        n_legal = len(legal)

        if step in DECISION_SCHEDULE:
            # The exact input the search gives the net at its ROOT: the flat 208-dim
            # observation, and the 64x84 feature block with the legal actions first.
            flat = flatten_obs({
                "my_hand":           obs["my_hand"],
                "table":             obs["table"],
                "my_team_captures":  obs["my_team_captures"],
                "opp_team_captures": obs["opp_team_captures"],
                "last_placed":       obs["last_placed"],
                "my_team_score":     obs["my_team_score"],
                "opp_team_score":    obs["opp_team_score"],
                "current_player":    obs["current_player"],
                "deck_remaining":    obs["deck_remaining"],
                "plays_this_deal":   obs["plays_this_deal"],
            }).astype(np.float32)

            with torch.no_grad():
                logits, value = net(
                    torch.as_tensor(flat)[None, None, :],
                    torch.zeros(1, 1, ACTION_FEAT_DIM),
                    torch.zeros(1, 1, dtype=torch.bool),
                    torch.as_tensor(feats[None], dtype=torch.float32),
                    torch.as_tensor(mask[None], dtype=torch.bool),
                )
            logits = logits[0].numpy()
            decisions.append({
                "name": f"step{step}_p{player}",
                "step": step,
                "player": player,
                "n_legal": n_legal,
                "obs": [float(x) for x in flat],
                "action_features": [float(x) for x in feats.reshape(-1)],
                "action_mask": [float(x) for x in mask.astype(np.float32)],
                "expected_slot": int(np.argmax(logits[:n_legal])),
                "expected_value": float(value[0]),
                "expected_logits": [float(x) for x in logits],
            })

        # Random legal play: the fixture only needs the POSITIONS to be real, and random
        # play reaches states (empty table, chained captures) faster than one policy would.
        slot = int(rng.integers(n_legal))
        tracker.record(player, obs, slot)
        env.step(slot)
        step += 1

    if not decisions:
        raise SystemExit("no decisions were captured — did the game end immediately?")

    fixture = {
        "pipeline": "ismcts",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "onnx": str(args.onnx),
        "value_target": args.value_target,
        "seed": args.seed,
        "obs_dim": 208,
        "action_feat_dim": ACTION_FEAT_DIM,
        "n_slots": MAX_ACTIONS,
        "lstm_hidden": 0,
        "recurrent_state": False,
        "decisions": decisions,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture))
    print(f"[fixture] wrote {out} ({len(decisions)} decisions, "
          f"checkpoint sha {sha256(ckpt)[:12]}…)")
    print(f"[fixture] value target: {args.value_target} | "
          f"expected values: {['%.3f' % d['expected_value'] for d in decisions]}")


if __name__ == "__main__":
    main()
