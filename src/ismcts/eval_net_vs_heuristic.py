"""
Evaluate a trained CuarentaNet checkpoint in a head-to-head match.

The net plays team 0 (seats 0 and 2). Team 1 is either:

  --opponent heuristic  (default) the HeuristicAgent: the reference opponent used
                        throughout this project. NOTE it was fixed recently (it used
                        to never play for a caída/limpia because of a `list == str`
                        bug), so it is a *stronger* opponent than the one earlier
                        checkpoints were measured against. Do not compare new
                        win rates with old numbers.
  --opponent raw        the same net with NO search: it plays the argmax of its own
                        policy head. This is the AlphaZero sanity check — if
                        "net + search" cannot beat "net + argmax policy", the search
                        is adding noise rather than strength, and self-play training
                        will degrade the policy instead of improving it.

Both sides use identical search/network settings; only the opponent differs.
Games run in worker processes, so all settings are passed explicitly (a spawned
worker does not see this process's globals).

Usage:
    uv run eval_net_vs_heuristic.py checkpoints/best.pt 100 --n-iter 200
    uv run eval_net_vs_heuristic.py checkpoints/best.pt 100 --opponent raw
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import timedelta
from multiprocessing import Pool
from typing import List

import numpy as np
from tqdm import tqdm


def play_one(args) -> dict:
    seed, ckpt_path, cfg, opponent = args

    import torch
    torch.set_num_threads(1)

    from env import CuarentaEnv, TEAMS
    from heuristic_agent import HeuristicAgent
    from history import HistoryTracker
    from network import CuarentaNet
    from network_ismcts_1 import NetworkISMCTS

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    net = CuarentaNet()
    CuarentaNet.load_checkpoint(net, ckpt_path)   # clear error on old checkpoints
    net.eval()

    tracker = HistoryTracker(env)
    searcher = NetworkISMCTS(
        env, net, tracker,
        n_iter=cfg["n_iter"],
        max_history_len=cfg["max_history_len"],
        leaf_value_mode=cfg["leaf_value_mode"],
        max_rollout_len=cfg["max_rollout_len"],
        rng=random.Random(seed * 2 + 1),
        dirichlet_eps=0.0,        # no exploration noise at eval
        device="cpu",
    )
    heuristic = HeuristicAgent(seed=seed * 2 + 2)

    steps = 0
    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()

        if TEAMS[p] == 0:
            # ---- our agent: search over visit counts ----
            visits = searcher.search(root_player=p)
            a = (max(visits, key=visits.get) if sum(visits.values()) > 0
                 else random.randrange(len(legal)))
        elif opponent == "heuristic":
            a = legal.index(heuristic.act(env, p))
        else:
            # ---- "raw" opponent: argmax of the net's policy head, no search ----
            # `_eval_root` is exactly the network call the search uses for its root
            # priors, so this is the same policy minus the lookahead.
            priors, _value = searcher._eval_root(p, legal)
            a = int(np.argmax(priors))

        tracker.record(p, obs, a)
        env.step(a)
        steps += 1

    return {
        "winner": env.state.winner,
        "score_team0": env.state.scores[0],
        "score_team1": env.state.scores[1],
        "steps": steps,
    }


def worker(args):
    return play_one(args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to a CuarentaNet .pt file")
    parser.add_argument("n_games", type=int, help="Number of games to play")
    parser.add_argument("--n-iter", type=int, default=200,
                        help="ISMCTS simulations per move (default: 200, same as training)")
    parser.add_argument("--max-rollout-len", type=int, default=512,
                        help="Only used with --leaf-value rollout")
    parser.add_argument("--max-history-len", type=int, default=32)
    parser.add_argument("--leaf-value", type=str, default="net",
                        choices=["net", "rollout"],
                        help="How leaf values are computed (default: net value head)")
    parser.add_argument("--opponent", type=str, default="heuristic",
                        choices=["heuristic", "raw"],
                        help="heuristic = HeuristicAgent, raw = same net without search")
    parser.add_argument("--seed-base", type=int, default=50000)
    args = parser.parse_args()

    cfg = {
        "n_iter": args.n_iter,
        "max_history_len": args.max_history_len,
        "leaf_value_mode": args.leaf_value,
        "max_rollout_len": args.max_rollout_len,
    }

    task_args = [(args.seed_base + i, args.checkpoint, cfg, args.opponent)
                 for i in range(args.n_games)]

    start = time.perf_counter()
    with Pool() as pool:
        results: List[dict] = list(tqdm(
            pool.imap(worker, task_args),
            total=args.n_games,
            desc="games",
        ))
    elapsed = time.perf_counter() - start

    wins_t0 = sum(1 for r in results if r["winner"] == 0)
    wins_t1 = sum(1 for r in results if r["winner"] == 1)
    score_t0 = np.mean([r["score_team0"] for r in results])
    score_t1 = np.mean([r["score_team1"] for r in results])
    steps_avg = np.mean([r["steps"] for r in results])

    # Standard error of a win rate: sqrt(p(1-p)/n). Useful because 30 games gives
    # roughly ±9%, which is why single evaluations looked so noisy.
    p = wins_t0 / max(args.n_games, 1)
    stderr = (p * (1 - p) / max(args.n_games, 1)) ** 0.5

    print()
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Games: {args.n_games}   n_iter: {args.n_iter}   "
          f"leaf_value: {args.leaf_value}   opponent: {args.opponent}")
    print(f"Net (team 0) wins:       {wins_t0}/{args.n_games} = {p:.1%} (±{stderr:.1%} se)")
    print(f"Opponent (team 1) wins:  {wins_t1}/{args.n_games} = {wins_t1/args.n_games:.1%}")
    print(f"Avg score: net {score_t0:.2f} - opponent {score_t1:.2f}")
    print(f"Avg game length: {steps_avg:.1f} plays")
    print(f"Time: {timedelta(seconds=elapsed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
