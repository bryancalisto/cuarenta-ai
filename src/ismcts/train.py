"""
Training loop for the Cuarenta net.

Simple supervised learning on self-play data (Expert Iteration style):

  for iteration in range(N_ITERS):
      1. Generate self-play games with the current net.
      2. Train the net to match (a) the visit distribution and (b) the outcome.
      3. Evaluate vs a reference opponent (random / previous net).
      4. Save checkpoint, log to W&B.

This is not PPO. It's cross-entropy for policy + MSE for value. Once this
loop is converging, we can consider switching to PPO.
"""

from __future__ import annotations

import os
import subprocess
import argparse
from pathlib import Path
import random
import time
from datetime import timedelta
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
import dotenv

from env import CuarentaEnv
from network import CuarentaNet
from selfplay import play_one_game, Decision
from dataset import CuarentaDataset
from network_ismcts_1 import NetworkISMCTS

dotenv.load_dotenv() # to load wandb token

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_ITERATIONS = 3              # outer loop iterations
# GAMES_PER_ITER = 40            # self-play games per iteration (bump if you can afford it)
GAMES_PER_ITER = 8            # self-play games per iteration (bump if you can afford it)
# N_SEARCH_ITER = 100            # ISMCTS sims per decision during self-play
# N_SEARCH_ITER = 50            # ISMCTS sims per decision during self-play
N_SEARCH_ITER = 100            # ISMCTS sims per decision during self-play
MAX_ROLLOUT_LEN = 80           # short rollout for speed; bump to 200 for stronger teacher
EPOCHS_PER_ITER = 8            # training epochs per iteration
BATCH_SIZE = 64
# LR = 1e-3
LR = 2.5e-3
T_MAX = 32
CKPT_DIR = "checkpoints"

# device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")


os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_games(net: CuarentaNet, n_games: int, base_seed: int) -> List[Decision]:
    all_decisions: List[Decision] = []
    for i in tqdm(range(n_games), desc="self-play games", leave=False):
        decisions = play_one_game(
            net,
            n_iter=N_SEARCH_ITER,
            max_rollout_len=MAX_ROLLOUT_LEN,
            seed=base_seed + i,
            temperature=1.0,
            max_history_len=T_MAX,
            device=DEVICE,
        )
        all_decisions.extend(decisions)
    return all_decisions


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(net, loader, opt, device):
    net.train()
    total_loss = 0.0
    total_p = 0.0
    total_v = 0.0
    n_batches = 0

    for (obs, act, am, feats, mask, pi_target, v_target) in loader:
        obs = obs.to(device)
        act = act.to(device)
        am = am.to(device)
        feats = feats.to(device)
        mask = mask.to(device)
        pi_target = pi_target.to(device)
        v_target = v_target.to(device)

        logits, value, _ = net(obs, act, am, feats, mask)

        # Policy loss: cross-entropy between softmax(logits) and pi_target.
        # We use log_softmax so we can ignore -1e9 masked slots cleanly.
        logp = torch.log_softmax(logits, dim=-1)
        # Safety: clamp pi_target to avoid log(0). Add tiny epsilon so masked
        # slots with pi=0 contribute 0.
        policy_loss = -(pi_target * logp).sum(dim=-1).mean()

        # Value loss: MSE against outcome (±1).
        value_loss = ((value - v_target) ** 2).mean()

        loss = policy_loss + value_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()
        total_p += policy_loss.item()
        total_v += value_loss.item()
        n_batches += 1

    return total_loss / n_batches, total_p / n_batches, total_v / n_batches


# ---------------------------------------------------------------------------
# Evaluation (vs random and vs rollout)
# ---------------------------------------------------------------------------

def eval_vs_random(net: CuarentaNet, n_games: int = 50, seed_base: int = 10000) -> float:
    """
    Play n_games where the net plays seats {0, 2} and random plays {1, 3}.
    Returns winrate for the net's team.
    """
    from env import TEAMS
    from rules import legal_actions_from_state
    from history import HistoryTracker
    from network_ismcts_1 import NetworkISMCTS
    from history import flatten_obs

    net.eval()
    wins = 0

    for g in range(n_games):
        env = CuarentaEnv(seed=seed_base + g)
        env.reset(seed=seed_base + g)
        tracker = HistoryTracker(env)
        searcher = NetworkISMCTS(
            env, net, tracker,
            n_iter=50,               # fewer sims for eval
            max_rollout_len=MAX_ROLLOUT_LEN,
            rng=random.Random(seed_base + g),
            dirichlet_eps=0.0,       # no noise at eval
            device=DEVICE,
        )

        while not env.state.done:
            p = env.state.current_player
            obs = env.observe(p)

            if p in (0, 2):
                visits = searcher.search(root_player=p)
                if sum(visits.values()) == 0:
                    legal = env.legal_actions()
                    a = random.randrange(len(legal))
                else:
                    a = max(visits, key=visits.get)
            else:
                legal = env.legal_actions()
                a = random.randrange(len(legal))

            tracker.record(p, obs, a)
            _, _, done, _, info = env.step(a)

        if info["winner"] == 0:
            wins += 1

    return wins / n_games

def eval_vs_heuristic_subprocess(ckpt_path: str, n_games: int = 30, n_iter: int = 100) -> float:
    """Run eval_net_vs_heuristic.py as a subprocess and parse the winrate."""
    script = Path(__file__).parent / "eval_net_vs_heuristic.py"
    result = subprocess.run(
        ["python", str(script), ckpt_path, str(n_games),
         "--n-iter", str(n_iter), "--leaf-value", "v1"],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        print(f"[eval_vs_heuristic] subprocess failed:\n{result.stderr}")
        return float("nan")
    for line in result.stdout.splitlines():
        if "Net (team 0) wins" in line:
            pct = float(line.split("=")[-1].strip().rstrip("%"))
            return pct / 100.0
    print(f"[eval_vs_heuristic] could not parse output:\n{result.stdout}")
    return float("nan")


def _eval_worker(args):
    """Play one eval game: net (team 0) vs rollout-ISMCTS (team 1)."""
    seed, state_dict, n_iter = args
    import torch
    torch.set_num_threads(1)

    from env import CuarentaEnv
    from network import CuarentaNet
    from history import HistoryTracker
    from network_ismcts_1 import NetworkISMCTS
    from ismcts import ISMCTS as RolloutISMCTS
    import random as _random

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    net_local = CuarentaNet()
    net_local.load_state_dict(state_dict)
    net_local.eval()

    tracker = HistoryTracker(env)
    net_searcher = NetworkISMCTS(
        env, net_local, tracker,
        n_iter=n_iter, max_rollout_len=200,
        rng=_random.Random(seed * 2 + 1), dirichlet_eps=0.0,
        device="cpu",                   # workers use CPU
    )
    rollout_searcher = RolloutISMCTS(
        env, n_iter=n_iter, rollout_policy="greedy",
        rng=_random.Random(seed * 2 + 2),
    )

    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        if p in (0, 2):
            visits = net_searcher.search()
        else:
            visits = rollout_searcher.search()
        legal = env.legal_actions()
        if sum(visits.values()) == 0:
            a = _random.randrange(len(legal))
        else:
            a = max(visits, key=visits.get)
        tracker.record(p, obs, a)
        _, _, _, _, info = env.step(a)

    return info["winner"]


def eval_vs_rollout(net, n_games=30, seed_base=20000, n_iter=100):
    """Net (team 0) vs rollout-ISMCTS (team 1). Returns winrate for team 0."""
    from multiprocessing import Pool
    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    args = [(seed_base + i, state_dict, n_iter) for i in range(n_games)]
    with Pool() as pool:
        results = pool.map(_eval_worker, args)
    wins = sum(1 for w in results if w == 0)
    return wins / n_games


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Optional path to a checkpoint to initialize from.")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--games-per-iter", type=int, default=GAMES_PER_ITER)
    parser.add_argument("--n-search-iter", type=int, default=N_SEARCH_ITER)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if not args.no_wandb:
        wandb.init(
            project="cuarenta-ai",
            config={
                "n_iterations": args.iterations,
                "games_per_iter": args.games_per_iter,
                "n_search_iter": args.n_search_iter,
                "max_rollout_len": MAX_ROLLOUT_LEN,
                "epochs_per_iter": EPOCHS_PER_ITER,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "t_max": T_MAX,
                "init_checkpoint": args.init_checkpoint,
            },
        )

    net = CuarentaNet().to(DEVICE)

    # Initialize from a checkpoint if requested.
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        net.load_state_dict(state)
        print(f"Initialized net from {args.init_checkpoint}")
    else:
        print("Starting from random initialization")

    opt = torch.optim.Adam(net.parameters(), lr=LR)

    best_winrate = -1.0
    start = time.perf_counter()
    decisions_buffer = []
    REPLAY_ITERS = 5


    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration} ===")

        # 1. Self-play
        t0 = time.perf_counter()
        decisions = collect_games(net, args.games_per_iter, base_seed=iteration * 1000)
        t_collect = time.perf_counter() - t0
        print(f"  collected {len(decisions)} decisions in {timedelta(seconds=t_collect)}")

        decisions_buffer.extend(decisions)
        # Keep only the most recent REPLAY_ITERS worth
        # Estimate as: REPLAY_ITERS * len(decisions) * 1.2 (safety margin)
        max_buffer = int(REPLAY_ITERS * len(decisions) * 1.5)
        if len(decisions_buffer) > max_buffer:
            decisions_buffer = decisions_buffer[-max_buffer:]

        # 2. Train
        # ds = CuarentaDataset(decisions, t_max=T_MAX)
        # loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        # Train on the full buffer, not just the current iteration
        ds = CuarentaDataset(decisions_buffer, t_max=T_MAX)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        for epoch in range(EPOCHS_PER_ITER):
            loss, pl, vl = train_one_epoch(net, loader, opt, DEVICE)
            if not args.no_wandb:
                wandb.log({"iter": iteration, "epoch": epoch,
                           "loss": loss, "policy_loss": pl, "value_loss": vl})
            print(f"  epoch {epoch}: loss={loss:.4f} policy={pl:.4f} value={vl:.4f}")

        
        # 3. Save
        ckpt_path = os.path.join(CKPT_DIR, f"cuarenta_iter{iteration:03d}.pt")
        torch.save(net.state_dict(), ckpt_path)
        print(f"  saved {ckpt_path}")

        # 4. Eval vs heuristic every 2 iterations
        if iteration % 2 == 0:
            wr_h = eval_vs_heuristic_subprocess(ckpt_path, n_games=30, n_iter=100)
            print(f"  eval vs heuristic: {wr_h:.2%}")
            if not args.no_wandb:
                wandb.log({"iter": iteration, "winrate_vs_heuristic": wr_h})
        # 5. Eval vs rollout every 5 iterations
        if iteration % 5 == 0:
            wr = eval_vs_rollout(net, n_games=30, seed_base=20000 + iteration * 100)
            print(f"  eval vs rollout-ISMCTS: {wr:.2%}")
            if not args.no_wandb:
                wandb.log({"iter": iteration, "winrate_vs_rollout": wr})


    print(f"\nTotal time: {timedelta(seconds=time.perf_counter() - start)}")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()