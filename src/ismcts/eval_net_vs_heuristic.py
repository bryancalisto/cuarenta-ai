"""
Evaluate a trained CuarentaNet checkpoint in a head-to-head match.

Same harness as the trainer's own gate: team 0 is the checkpoint, the opponent is
HeuristicAgent, the search is greedy with no Dirichlet noise, and the per-game RNG follows
`random.Random(seed)` like `train.py: _make_search`. With `--seed-base 50000` and
`--value-target` matching the checkpoint, this reproduces the trainer's `eval/heuristic`
number EXACTLY for the same snapshot and game count — which is what makes it useful for
answering "why do my two numbers disagree?".

The report carries the same per-game fields as the trainer's `eval/heuristic_*` keys —
win rate, score, RETURN (the dense env reward, per team), caídas and limpias, and game
length — so a one-off measurement can be read next to (or quoted against) the training
curves. The counters are the trainer's own: +1 per captured card, +2 caída, +2 limpia, the
cartón points and the win bonus, and every game asserts that the return decomposes into
exactly those terms before it is counted.

The value target must be right (`--value-target`, else the checkpoint's sidecar): in return
mode the search accumulates the dense reward, in win mode it ignores it and backs up ±1 game
results.

The net plays team 0 (seats 0 and 2). Team 1 is either:

  --opponent heuristic  (default) the HeuristicAgent: the reference opponent used
                        throughout this project. Note it plays for a caída/limpia now, so it is a *stronger*
                        opponent than the one older numbers were taken against. Do
                        not compare new win rates with old ones.
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

# Bare sibling import (this directory is a script directory, see AGENTS.md).
from network import VALUE_TARGETS


def play_one(args) -> dict:
    seed, ckpt_path, cfg, opponent = args
    agent = cfg.get("agent", "search")

    import torch
    torch.set_num_threads(1)

    from env import WIN_BONUS, CuarentaEnv, TEAMS
    from heuristic_agent import HeuristicAgent
    from history import HistoryTracker
    from network import CuarentaNet, VALUE_TARGETS
    from network_ismcts_1 import NetworkISMCTS
    from utils import PointsReasons, has_reason

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    # The value target must match the weights, or the search runs as a different agent:
    # `resolve_value_target` takes it from --value-target, else the checkpoint's sidecar,
    # else 'win' with a warning. See CuarentaNet.from_checkpoint.
    net = CuarentaNet.from_checkpoint(ckpt_path, value_target=cfg.get("value_target"))
    net.eval()

    tracker = HistoryTracker(env)
    searcher = NetworkISMCTS(
        env, net, tracker,
        n_iter=cfg["n_iter"],
        max_history_len=cfg["max_history_len"],
        leaf_value_mode=cfg["leaf_value_mode"],
        max_rollout_len=cfg["max_rollout_len"],
        # SAME per-game RNG convention as the trainer's evaluation workers
        # (`train.py: _make_search(..., seed)` -> random.Random(seed)). The searcher's RNG
        # drives the determinization sampling, so a different convention means different
        # realized games even for the same environment seed, and the two harnesses would
        # then disagree about the same snapshot.
        rng=random.Random(seed),
        dirichlet_eps=0.0,        # no exploration noise at eval
        device="cpu",
    )
    heuristic = HeuristicAgent(seed=seed * 2 + 2)

    # Per-team bookkeeping, the same one the trainer's own `_play_match` keeps (see
    # `train.py: match_metrics`), so these numbers are comparable with the `eval/*` curves
    # the training runs log. The cartón and the captured-card count are only needed for the
    # identity check below: the return already contains them.
    returns = [0.0, 0.0]
    caidas = [0, 0]
    limpias = [0, 0]
    captures = [0, 0]
    carton = [0, 0]
    steps = 0
    while not env.state.done:
        p = env.state.current_player
        obs = env.observe(p)
        legal = env.legal_actions()

        if TEAMS[p] == 0:
            if agent == "raw":
                # ---- our agent WITHOUT the search: argmax of the policy head ----
                # This is exactly what the Go backend does (one ONNX forward pass, argmax
                # over the legal logits), so it answers "what will the served AI feel
                # like?" — and it is the same policy the trainer's `teacher_edge` metric
                # measures the search against.
                priors, _value = searcher._eval_root(p, legal)
                a = int(np.argmax(priors))
            else:
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

        # `legal[a]` IS the action the env applies, so the capture count is exact; the
        # capture piles cannot be read back after a cartón clears them.
        played = legal[a]
        _, reward, _, _, info = env.step(a)

        team = TEAMS[p]
        returns[team] += float(reward)
        if played.capture:
            captures[team] += len(played.capture)
        # caída and limpia are mutually exclusive in the env's own reward (if/elif), so
        # mirror that here to keep the identity below exact.
        reasons = info.get("reasons") or []
        if has_reason(reasons, PointsReasons.CAIDA):
            caidas[team] += 1
        elif has_reason(reasons, PointsReasons.LIMPIA):
            limpias[team] += 1
        # The cartón is paid to the acting team on the move that ends a deal.
        carton[team] += int(info.get("carton_points") or 0)
        steps += 1

    # The env's reward is dense and fully explained by the counters above, so this is a
    # guard that the harness counts what it reports: +1 per captured card, +2 caída,
    # +2 limpia, the cartón points, WIN_BONUS for the win.
    for team in (0, 1):
        won = 1 if env.state.winner == team else 0
        expected = (captures[team] + 2 * caidas[team] + 2 * limpias[team]
                    + carton[team] + WIN_BONUS * won)
        if abs(returns[team] - expected) > 1e-6:
            raise AssertionError(
                f"seed {seed}: team {team} accumulated return {returns[team]} but "
                f"captures + 2*caidas + 2*limpias + carton + WIN_BONUS*won = {expected} "
                f"(captures={captures[team]}, caidas={caidas[team]}, "
                f"limpias={limpias[team]}, carton={carton[team]}, won={won})")

    return {
        "winner": env.state.winner,
        "score_team0": env.state.scores[0],
        "score_team1": env.state.scores[1],
        # `_for` is the checkpoint's team (team 0), like every other harness here.
        "return_for": returns[0],
        "return_against": returns[1],
        "caidas_for": caidas[0],
        "caidas_against": caidas[1],
        "limpias_for": limpias[0],
        "limpias_against": limpias[1],
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
    parser.add_argument("--agent", type=str, default="search",
                        choices=["search", "raw"],
                        help="how OUR side plays: 'search' = ISMCTS (what the training "
                             "evaluations measure), 'raw' = argmax of the policy head with "
                             "no search, which is exactly what the Go backend serves. The "
                             "gap between the two is the value of searching.")
    parser.add_argument("--opponent", type=str, default="heuristic",
                        choices=["heuristic", "raw"],
                        help="heuristic = HeuristicAgent, raw = same net without search")
    parser.add_argument("--seed-base", type=int, default=50000)
    parser.add_argument("--value-target", type=str, default=None,
                        choices=list(VALUE_TARGETS),
                        help="value target the checkpoint was trained with; defaults to the "
                             "value recorded in its .json sidecar, else 'win'. Passing the "
                             "wrong one evaluates a different agent: in return mode the "
                             "search accumulates the dense reward, in win mode it ignores "
                             "it and backs up ±1 game results")
    args = parser.parse_args()

    cfg = {
        "n_iter": args.n_iter,
        "max_history_len": args.max_history_len,
        "leaf_value_mode": args.leaf_value,
        "max_rollout_len": args.max_rollout_len,
        "value_target": args.value_target,
        "agent": args.agent,
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

    def m(key: str) -> float:
        return float(np.mean([r[key] for r in results]))

    # Standard error of a win rate: sqrt(p(1-p)/n). At 30 games that is roughly ±9%, so
    # a single small evaluation cannot separate two close nets.
    p = wins_t0 / max(args.n_games, 1)
    stderr = (p * (1 - p) / max(args.n_games, 1)) ** 0.5

    print()
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"agent: {args.agent}")
    print(f"Games: {args.n_games}   n_iter: {args.n_iter}   "
          f"leaf_value: {args.leaf_value}   opponent: {args.opponent}")
    print(f"Net (team 0) wins:       {wins_t0}/{args.n_games} = {p:.1%} (±{stderr:.1%} se)")
    print(f"Opponent (team 1) wins:  {wins_t1}/{args.n_games} = {wins_t1/args.n_games:.1%}")
    print(f"Avg score: net {score_t0:.2f} - opponent {score_t1:.2f}")
    # The per-game numbers the cross-model tables use, from the net's point of view
    # (`for` = team 0, `against` = the opponent). Same definitions as the trainer's
    # `eval/*_return_*`, `_caidas_*` and `_limpias_*` keys.
    print(f"Avg return: net {m('return_for'):.2f} - opponent {m('return_against'):.2f}"
          f"   (dense env reward)")
    print(f"Avg caídas: net {m('caidas_for'):.2f} - opponent {m('caidas_against'):.2f}")
    print(f"Avg limpias: net {m('limpias_for'):.2f} - opponent {m('limpias_against'):.2f}")
    print(f"Avg game length: {steps_avg:.1f} plays")
    print(f"Time: {timedelta(seconds=elapsed)} ({elapsed / max(args.n_games, 1):.2f} s/game)")
    print("=" * 60)


if __name__ == "__main__":
    main()
