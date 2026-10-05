"""

Evaluate a trained CuarentaNet checkpoint against the HeuristicAgent.

The net plays team 0 (seats 0 and 2). The heuristic plays team 1 (seats 1, 3).

Uses multiprocessing so each game runs in its own process.
Reports winrate, score stats, and game length.

Usage:
    uv run eval_net_vs_heuristic.py checkpoints/cuarenta_iter012.pt 100
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import timedelta
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from env import CuarentaEnv, TEAMS
from history import HistoryTracker
from network import CuarentaNet
from network_ismcts_1 import NetworkISMCTS
from network_ismcts_v4 import NetworkISMCTS as NetworkISMCTS_net
from network_ismcts_v3 import NetworkISMCTS as NetworkISMCTS_v3
from heuristic_agent import HeuristicAgent


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_one(args) -> dict:
    seed, ckpt_path, n_iter, max_rollout_len, leaf_value = args

    import torch
    torch.set_num_threads(1)

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    net = CuarentaNet()
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    net.load_state_dict(state)
    net.eval()

    tracker = HistoryTracker(env)
    
    if leaf_value == "net":
        net_searcher = NetworkISMCTS_net(
            env, net, tracker,
            n_iter=n_iter, max_rollout_len=max_rollout_len,
            rng=random.Random(seed * 2 + 1),
            dirichlet_eps=0.0, device="cpu",
            leaf_value_mode="net",
        )
    elif leaf_value == "rollout":
        net_searcher = NetworkISMCTS(
            env, net, tracker,
            n_iter=n_iter, max_rollout_len=max_rollout_len,
            rng=random.Random(seed * 2 + 1),
            dirichlet_eps=0.0, device="cpu",
            leaf_value_mode="rollout",
        )
    else:  # "v1" for the original
        from network_ismcts_1 import NetworkISMCTS as NetworkISMCTS_v1
        net_searcher = NetworkISMCTS_v1(
            env, net, tracker,
            n_iter=n_iter, max_rollout_len=max_rollout_len,
            rng=random.Random(seed * 2 + 1),
            dirichlet_eps=0.0, device="cpu",
        )
    
    heuristic = HeuristicAgent(seed=seed * 2 + 2)

    steps = 0
    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)

        if p in (0, 2):
            # Net's turn
            visits = net_searcher.search()
            legal = env.legal_actions()
            if sum(visits.values()) == 0:
                a = random.randrange(len(legal))
            else:
                a = max(visits, key=visits.get)
            action = legal[a]
        else:
            # Heuristic's turn
            action = heuristic.act(env, p)

        # Record into tracker with the slot index of the chosen action
        legal_now = env.legal_actions()
        slot = legal_now.index(action)
        tracker.record(p, obs, slot)

        env.step(slot)
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
    parser.add_argument("checkpoint", type=str, help="Path to CuarentaNet .pt file")
    parser.add_argument("n_games", type=int, help="Number of games to play")
    parser.add_argument("--n-iter", type=int, default=100,
                        help="ISMCTS sims per move for the net (default: 100)")
    parser.add_argument("--max-rollout-len", type=int, default=200,
                        help="Rollout length at leaves (default: 200)")
    parser.add_argument("--seed-base", type=int, default=50000)
    parser.add_argument("--leaf-value", type=str, default="v1") # v1 | net | rollout
    args = parser.parse_args()

    task_args = [
        (args.seed_base + i, args.checkpoint, args.n_iter, args.max_rollout_len, args.leaf_value)
        for i in range(args.n_games)
    ]

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

    print()
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Games: {args.n_games}   n_iter: {args.n_iter}   max_rollout_len: {args.max_rollout_len}")
    print(f"Net (team 0) wins:       {wins_t0}/{args.n_games} = {wins_t0/args.n_games:.1%}")
    print(f"Heuristic (team 1) wins: {wins_t1}/{args.n_games} = {wins_t1/args.n_games:.1%}")
    print(f"Avg score: net {score_t0:.2f} - heuristic {score_t1:.2f}")
    print(f"Avg game length: {steps_avg:.1f} plays")
    print(f"Time: {timedelta(seconds=elapsed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

    """
    # 100 games, default search settings
    uv run src/ismcts/eval_net_vs_heuristic.py checkpoints/cuarenta_iter012.pt 100

    # Faster eval with fewer sims
    uv run src/ismcts/eval_net_vs_heuristic.py checkpoints/cuarenta_iter012.pt 100 --n-iter 50


============================================================
Checkpoint: checkpoints/cuarenta_iter012.pt
Games: 100   n_iter: 50   max_rollout_len: 200
Net (team 0) wins:       38/100 = 38.0%
Heuristic (team 1) wins: 62/100 = 62.0%
Avg score: net 27.74 - heuristic 36.42
Avg game length: 134.6 plays
Time: 0:01:45.012455
    
    ============================================================
Checkpoint: checkpoints/cuarenta_iter012.pt
Games: 100   n_iter: 200   max_rollout_len: 200
Net (team 0) wins:       48/100 = 48.0%
Heuristic (team 1) wins: 52/100 = 52.0%
Avg score: net 29.68 - heuristic 33.26
Avg game length: 131.7 plays
Time: 0:06:55.472011
    """
    