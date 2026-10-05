"""
Bootstrap CuarentaNet via imitation learning from the HeuristicAgent.

Phase 1 of the training plan:
  1. Generate N games where all four seats play HeuristicAgent.
  2. Record every decision as a training sample:
       (obs_seq, act_seq, act_mask, action_features, action_mask, one_hot_action, outcome)
  3. Train CuarentaNet to imitate the heuristic (cross-entropy) and predict
     game outcome (MSE).

This gives the net a competent starting policy. Subsequent self-play starts
from a strong baseline instead of random weights, which makes the whole
training loop dramatically more productive.

Usage:
    # Default: 2000 games, 8 epochs
    uv run src/ismcts/bootstrap.py

    # Faster iteration
    uv run src/ismcts/bootstrap.py --games 500 --epochs 5

    # Save to a specific path
    uv run src/ismcts/bootstrap.py --games 2000 --epochs 10 --out checkpoints/bootstrap_v1.pt

    # Skip W&B
    uv run src/ismcts/bootstrap.py --no-wandb
"""

from __future__ import annotations

import argparse
import os
import random
import time
from datetime import timedelta
from multiprocessing import Pool
from typing import List
import dotenv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from env import CuarentaEnv, TEAMS, NUM_PLAYERS
from heuristic_agent import HeuristicAgent
from history import HistoryTracker, flatten_obs
from network import CuarentaNet, N_ACTIONS, ACTION_FEAT_DIM
from dataset import CuarentaDataset
from selfplay import Decision


dotenv.load_dotenv()
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

T_MAX = 32
DEFAULT_GAMES = 2000
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def play_one_heuristic_game(seed: int) -> List[Decision]:
    """
    Play a full game with HeuristicAgent at all four seats.
    Return one Decision per decision point (from the acting player's POV).
    """
    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    agent = HeuristicAgent(seed=seed)
    tracker = HistoryTracker(env)
    decisions: List[Decision] = []

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()

        # Heuristic chooses an Action object.
        action = agent.act(env, p)
        slot = legal.index(action)

        # Build the history sequence for this decision. If empty (first decision
        # of the game), synthesize a length-1 obs from the current state.
        obs_seq, act_seq, act_mask, _ = tracker.sequence_for(p, max_len=T_MAX)
        if obs_seq.shape[0] == 0:
            obs_seq = flatten_obs(obs)[None, :]
            act_seq = np.zeros((1, N_ACTIONS), dtype=np.int64)
            act_mask = np.zeros((1,), dtype=bool)

        # One-hot target over the *slot* index.
        visits = np.zeros(N_ACTIONS, dtype=np.int64)
        visits[slot] = 1

        decisions.append(Decision(
            obs_seq=obs_seq.astype(np.float32),
            act_seq=act_seq.astype(np.float32),
            act_mask=act_mask.astype(bool),
            feats=obs["action_features"].astype(np.float32),
            mask=obs["action_mask"].astype(bool),
            visits=visits,
            team=TEAMS[p],
            player=p,
        ))

        tracker.record(p, obs, slot)
        env.step(slot)

    # Backfill outcome from the acting player's team POV.
    winner_team = env.state.winner
    for d in decisions:
        d.outcome = 1.0 if d.team == winner_team else -1.0

    return decisions


def _collect_worker(seed: int) -> List[Decision]:
    import torch
    torch.set_num_threads(1)
    return play_one_heuristic_game(seed)


def collect_heuristic_games(n_games: int, seed_base: int = 0) -> List[Decision]:
    """
    Parallel data collection. Each worker plays one game.
    """
    n_proc = os.cpu_count() or 1
    print(f"Collecting {n_games} heuristic games with {n_proc} workers...")

    t0 = time.perf_counter()
    with Pool(n_proc) as pool:
        results = list(tqdm(
            pool.imap(_collect_worker, range(seed_base, seed_base + n_games)),
            total=n_games,
            desc="games",
        ))

    decisions = [d for game in results for d in game]
    dt = time.perf_counter() - t0
    print(f"Collected {len(decisions)} decisions in {timedelta(seconds=dt)}")
    return decisions


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(net, loader, opt, device):
    net.train()
    total_loss = 0.0
    total_p = 0.0
    total_v = 0.0
    total_correct = 0
    total_samples = 0

    for (obs, act, am, feats, mask, pi_target, v_target) in loader:
        obs = obs.to(device)
        act = act.to(device)
        am = am.to(device)
        feats = feats.to(device)
        mask = mask.to(device)
        pi_target = pi_target.to(device)
        v_target = v_target.to(device)

        logits, value, _ = net(obs, act, am, feats, mask)

        logp = torch.log_softmax(logits, dim=-1)
        policy_loss = -(pi_target * logp).sum(dim=-1).mean()
        value_loss = ((value - v_target) ** 2).mean()
        loss = policy_loss + value_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        # Track accuracy: did we pick the heuristic's action?
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            target = pi_target.argmax(dim=-1)
            total_correct += (pred == target).sum().item()
            total_samples += obs.size(0)

        total_loss += loss.item()
        total_p += policy_loss.item()
        total_v += value_loss.item()

    n = len(loader)
    return (
        total_loss / n,
        total_p / n,
        total_v / n,
        total_correct / max(total_samples, 1),
    )


def train(
    decisions: List[Decision],
    *,
    out_path: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: str = "cpu",
):
    print(f"\nTraining on {len(decisions)} samples for {epochs} epochs on {device}")

    ds = CuarentaDataset(decisions, t_max=T_MAX)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    net = CuarentaNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        loss, pl, vl, acc = train_one_epoch(net, loader, opt, device)
        dt = time.perf_counter() - t0

        print(f"  epoch {epoch}: loss={loss:.4f} policy={pl:.4f} "
              f"value={vl:.4f} acc={acc:.3f} ({dt:.1f}s)")

        wandb.log({
            "bootstrap/epoch": epoch,
            "bootstrap/loss": loss,
            "bootstrap/policy_loss": pl,
            "bootstrap/value_loss": vl,
            "bootstrap/action_accuracy": acc,
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(net.state_dict(), out_path)
    print(f"\nSaved {out_path}")

    return net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="Number of heuristic games to generate")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--out", type=str, default="checkpoints/bootstrap.pt")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                        help="cpu | mps | cuda. Defaults to auto.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional path to a checkpoint to initialize the net from "
            "(e.g. checkpoints/bootstrap.pt). If not given, starts from random init.",
    )

    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Using device: {device}")

    if not args.no_wandb:
        wandb.init(
            project="cuarenta-ai",
            name="bootstrap",
            config={
                "games": args.games,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "t_max": T_MAX,
                "device": device,
            },
        )

    decisions = collect_heuristic_games(args.games, seed_base=args.seed_base)
    train(
        decisions,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

    """
    
 epoch 0: loss=1.3965 policy=0.6286 value=0.7679 acc=0.640 (157.5s)
  epoch 1: loss=1.0499 policy=0.6239 value=0.4260 acc=0.641 (156.9s)
  epoch 2: loss=0.8433 policy=0.6238 value=0.2196 acc=0.640 (158.6s)
  epoch 3: loss=0.8062 policy=0.6234 value=0.1828 acc=0.642 (164.1s)
  epoch 4: loss=0.7726 policy=0.6231 value=0.1494 acc=0.642 (157.6s)
  epoch 5: loss=0.7420 policy=0.6229 value=0.1191 acc=0.642 (160.6s)
  epoch 6: loss=0.7210 policy=0.6227 value=0.0983 acc=0.643 (160.1s)
  epoch 7: loss=0.7057 policy=0.6224 value=0.0833 acc=0.644 (162.1s)
  epoch 8: loss=0.6959 policy=0.6222 value=0.0737 acc=0.645 (159.5s)
  epoch 9: loss=0.6870 policy=0.6220 value=0.0650 acc=0.645 (158.1s)
  epoch 10: loss=0.6806 policy=0.6216 value=0.0589 acc=0.647 (156.4s)
  epoch 11: loss=0.6753 policy=0.6212 value=0.0540 acc=0.648 (160.8s)
  epoch 12: loss=0.6712 policy=0.6209 value=0.0503 acc=0.649 (156.5s)
  epoch 13: loss=0.6679 policy=0.6206 value=0.0473 acc=0.650 (159.2s)
  epoch 14: loss=0.6637 policy=0.6202 value=0.0436 acc=0.651 (156.4s)
  epoch 15: loss=0.6614 policy=0.6197 value=0.0418 acc=0.653 (155.3s)
  epoch 16: loss=0.6585 policy=0.6191 value=0.0394 acc=0.653 (157.2s)
  epoch 17: loss=0.6560 policy=0.6184 value=0.0376 acc=0.655 (157.1s)
  epoch 18: loss=0.6545 policy=0.6179 value=0.0366 acc=0.656 (158.2s)
  epoch 19: loss=0.6528 policy=0.6172 value=0.0357 acc=0.658 (156.8s) 

  
  Best bootstrap so far:
  
  Collecting 3000 heuristic games with 10 workers...
games: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [00:17<00:00, 167.94it/s]
Collected 385608 decisions in 0:00:17.989256

Training on 385608 samples for 12 epochs on mps
  epoch 0: loss=1.3679 policy=0.6339 value=0.7341 acc=0.641 (86.5s)
  epoch 1: loss=0.9048 policy=0.6240 value=0.2809 acc=0.641 (85.6s)
  epoch 2: loss=0.8210 policy=0.6235 value=0.1975 acc=0.642 (86.8s)
  epoch 3: loss=0.7940 policy=0.6232 value=0.1709 acc=0.642 (86.8s)
  epoch 4: loss=0.7662 policy=0.6227 value=0.1434 acc=0.643 (86.7s)
  epoch 5: loss=0.7384 policy=0.6225 value=0.1159 acc=0.643 (85.7s)
  epoch 6: loss=0.7139 policy=0.6222 value=0.0918 acc=0.644 (87.8s)
  epoch 7: loss=0.6927 policy=0.6220 value=0.0707 acc=0.645 (87.6s)
  epoch 8: loss=0.6804 policy=0.6215 value=0.0589 acc=0.646 (86.1s)
  epoch 9: loss=0.6696 policy=0.6212 value=0.0485 acc=0.647 (86.1s)
  epoch 10: loss=0.6619 policy=0.6209 value=0.0410 acc=0.647 (85.8s)
  epoch 11: loss=0.6567 policy=0.6205 value=0.0363 acc=0.648 (86.2s)

Saved checkpoints/bootstrap.pt
wandb: 
wandb: Run history:
wandb: bootstrap/action_accuracy ▁▁▂▂▃▃▄▅▅▇▇█
wandb:           bootstrap/epoch ▁▂▂▃▄▄▅▅▆▇▇█
wandb:            bootstrap/loss █▃▃▂▂▂▂▁▁▁▁▁
wandb:     bootstrap/policy_loss █▃▃▂▂▂▂▂▂▁▁▁
wandb:      bootstrap/value_loss █▃▃▂▂▂▂▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb: bootstrap/action_accuracy 0.64834
wandb:           bootstrap/epoch 11
wandb:            bootstrap/loss 0.65673
wandb:     bootstrap/policy_loss 0.62046
wandb:      bootstrap/value_loss 0.03627
wandb: 
wandb: 🚀 View run bootstrap at: https://wandb.ai/einecomadreja-stealth-corp/cuarenta-ai/runs/89rq0szw
wandb: ⭐️ View project at: https://wandb.ai/einecomadreja-stealth-corp/cuarenta-ai
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20261002_222106-89rq0szw/logs
(.venv) t1000@t1000s-MacBook-Pro 40-ai % uv run src/ismcts/eval_net_vs_heuristic.py checkpoints/bootstrap.pt 100 --n-iter 200 --leaf-value v1
games: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [05:38<00:00,  3.39s/it]

============================================================
Checkpoint: checkpoints/bootstrap.pt
Games: 100   n_iter: 200   max_rollout_len: 200
Net (team 0) wins:       65/100 = 65.0%
Heuristic (team 1) wins: 35/100 = 35.0%
Avg score: net 35.46 - heuristic 31.70
Avg game length: 133.9 plays
Time: 0:05:38.993344
    """
    