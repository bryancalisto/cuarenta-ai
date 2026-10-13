#!/usr/bin/env python3
"""
Write a JSON fixture that pins PyTorch's policy outputs, so the ONNX export can be
checked for numerical parity from another language (the Go backend, which has the only
onnxruntime on this machine).

Why this exists: the backend serves real players through ONNX, not through PyTorch. If
the export ever silently changed semantics -- the wrong observation order, a transposed
action-feature matrix, a missing mask, a value head read from the wrong output -- the Go
service would still "run" and still return a legal slot, just a *worse* one, and nothing
would complain. This fixture turns that class of bug into a test failure.

Usage:
    python3 scripts/onnx_parity_fixture.py
        --checkpoint checkpoints-ppo/bootstrap.pt
        --out backend/internal/inference/testdata/policy_fixture.json

Then, from `backend/`:
    go test ./internal/inference/ -run TestONNXMatchesPyTorch -v
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "ppo"))

from env import CuarentaEnv, action_features_matrix          # noqa: E402
from utils import flatten_obs                                # noqa: E402
from networks import load_actor_critic                       # noqa: E402

# Decisions are sampled at a few points of one game so the fixture covers more than a
# single opening position (an empty table, a non-empty table, and a later deal).
DECISION_SCHEDULE = [0, 7, 15]


def build_fixture(checkpoint: str, seed: int, device: torch.device):
    net = load_actor_critic(checkpoint, device)
    env = CuarentaEnv(seed=seed)
    env.reset()

    fixtures = []
    steps_done = 0
    wanted = list(DECISION_SCHEDULE)

    while wanted and not env.state.done:
        player = env.state.current_player
        legal = env.legal_actions()

        if steps_done == wanted[0]:
            wanted.pop(0)
            obs = env.observe(player)
            obs_flat = flatten_obs(obs)
            feats, mask = action_features_matrix(legal, env.state)

            obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
            feats_t = torch.as_tensor(feats, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, device=device).unsqueeze(0)
            with torch.no_grad():
                # hidden=None means a zero-initialised LSTM state, which is exactly what
                # the Go service feeds on a game's first decision.
                logits, value, _ = net(obs_t, feats_t, mask_t, None)

            logits = logits[0].detach().cpu().numpy()
            n_legal = len(legal)
            # The model writes a large negative number on illegal slots; the choice is an
            # argmax restricted to the caller's own legal slots, which is what SelectSlot
            # does on the Go side too.
            expected_slot = int(np.argmax(logits[:n_legal]))

            fixtures.append({
                "name": f"step{steps_done}_p{player}",
                "step": steps_done,
                "player": player,
                "n_legal": n_legal,
                "obs": [float(x) for x in obs_flat],
                "action_features": [float(x) for x in np.asarray(feats).reshape(-1)],
                "action_mask": [float(x) for x in np.asarray(mask).reshape(-1)],
                "expected_slot": expected_slot,
                "expected_value": float(value.item()),
                "expected_logits": [float(x) for x in logits],
            })

        # Deterministic replay: always take the first legal action so the fixture does not
        # depend on the RNG of the action choice, only on the seeded deal.
        env.step(legal[0])
        steps_done += 1

    if len(fixtures) != len(DECISION_SCHEDULE):
        raise SystemExit(
            f"only captured {len(fixtures)}/{len(DECISION_SCHEDULE)} decisions; "
            "the game ended early -- lower DECISION_SCHEDULE"
        )
    return net, fixtures


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints-ppo" / "bootstrap.pt"))
    ap.add_argument("--out", default=str(REPO / "backend" / "internal" / "inference"
                                        / "testdata" / "policy_fixture.json"))
    ap.add_argument("--onnx", default=str(REPO / "backend" / "models" / "ppo_bootstrap.onnx"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    net, fixtures = build_fixture(args.checkpoint, args.seed, device)

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        # The Go test compares this with the ONNX sidecar's source_checkpoint_sha256, so a
        # model re-exported from newer weights cannot be "validated" by this stale fixture.
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "onnx": Path(args.onnx).name,
        "seed": args.seed,
        "obs_dim": net.obs_dim,
        "action_feat_dim": net.action_feat_dim,
        "n_slots": net.n_slots,
        "lstm_hidden": net.lstm_hidden if net.use_lstm else 0,
        "decisions": fixtures,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))

    # A fixture whose expected slots are all 0 cannot catch a "always return slot 0"
    # bug, so refuse to write one.
    slots = [f["expected_slot"] for f in fixtures]
    if all(s == 0 for s in slots):
        raise SystemExit(f"degenerate fixture: every expected slot is 0 {slots}")

    print(f"[fixture] wrote {out}")
    for f in fixtures:
        print(f"[fixture]   {f['name']:12s} n_legal={f['n_legal']} "
              f"slot={f['expected_slot']} value={f['expected_value']:+.6f}")


if __name__ == "__main__":
    main()
