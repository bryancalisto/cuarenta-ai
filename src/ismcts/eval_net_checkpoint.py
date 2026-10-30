#!/usr/bin/env python3
"""
Evaluate two ISMCTS checkpoints against each other, head to head.

WHAT THIS MEASURES
------------------
Two CuarentaNet checkpoints, each driving its own `NetworkISMCTS` search, in the same
game. This is the comparison the trainer cannot make for you: it gates each new snapshot
against the previous best, but it never asks whether checkpoint A from one run beats
checkpoint B from another -- a run with a different value target, a different number of
iterations, or an older recipe. Same deal, same rules, same search budget unless you set
one.

Both sides are the SAME pipeline, so unlike the cross-pipeline evaluators (`src/eval_*`)
this needs no second process and no mirrored game: one env, two searchers, two history
trackers. Each tracker records every turn, whoever played it, because a `NetworkISMCTS`
search reads the whole public history, and a tracker that skipped the opponent's turns
would hide half the game from the search.

VALUE TARGETS: THE ONE SETTING THAT IS NOT IN THE WEIGHTS
--------------------------------------------------------
Win mode backs up ±1 game results; return mode backs up the dense shaped reward. The head
differs only by an activation, so a return-mode checkpoint loads into a win-mode search
without a word and quietly plays as a different agent. Each side therefore resolves its
own mode: from `--value-target`, else from the `.json` sidecar the trainer writes beside
every checkpoint, else 'win' with a warning. `--value-target` accepts one value for both
sides ('return'), or a pair in checkpoint order ('return,win') -- any mix is fine, which
is what makes comparing a win-mode run against a return-mode run meaningful.

SEATS AND THE PAIRED DESIGN
---------------------------
A controls a TEAM (seats 0+2 or 1+3) and so does B. Every deal is played TWICE with the
roles swapped, so the two games share a shuffle and only the side controls differ.
Cuarenta deals decide a lot, so same-deal pairing is the cheapest way to see a real
difference through the noise, and it turns the per-team split in the report into a direct
read on any first-team advantage. Metrics are reported from A's point of view, matching the
`_for` / `_against` convention of every other harness in this project.

Both sides play greedily (the most-visited root child, `dirichlet_eps=0`), and each game's
two searches draw their determinizations from their own seeded `random.Random`, so a
(checkpoints, seed) pair always reproduces the same game.

REWARD IDENTITY SELF-CHECK
--------------------------
After every game the harness proves it counted what the env paid: each team's accumulated
reward must equal `captures + 2*caidas + 2*limpias + carton + WIN_BONUS*won` exactly. A
failed identity means the report does not measure the game, so that game is reported as
failed rather than silently included.

Usage:
    uv run src/ismcts/eval_net_checkpoint.py \\
        --checkpoint-a checkpoints-return/best.pt --checkpoint-b checkpoints/best.pt \\
        --games 20 --value-target return,win

    # how much of a difference is the search budget?
    uv run src/ismcts/eval_net_checkpoint.py --checkpoint-a A.pt --checkpoint-b B.pt \\
        --games 20 --sweep 32,128,400

Timing: both sides search, so a game costs two searches -- roughly `--n-iter` simulations
per turn for each side. `--workers` (one game per process) is the knob that matters above
a few dozen games.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

def play_one(args) -> dict:
    """Play one game; `a_team` is the team checkpoint A controls (0 or 1)."""
    seed, ckpt_a, ckpt_b, cfg, a_team = args

    import torch

    torch.set_num_threads(1)
    from env import WIN_BONUS, CuarentaEnv, TEAMS
    from history import HistoryTracker
    from network import CuarentaNet
    from network_ismcts_1 import NetworkISMCTS
    from utils import PointsReasons, has_reason

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    # Each checkpoint resolves its own value target (`from_checkpoint` takes
    # --value-target, else the sidecar, and warns when it has neither). The mode is NOT
    # part of the state_dict, and a search built in the wrong mode plays another agent.
    checkpoints = {a_team: ckpt_a, 1 - a_team: ckpt_b}
    targets = {a_team: cfg["value_target_a"], 1 - a_team: cfg["value_target_b"]}

    # Team -> (net, tracker, searcher). The two determinization streams are separate and
    # seeded, so a game is reproducible and neither side inherits the other's luck.
    nets, trackers, searchers = {}, {}, {}
    for team, team_seed in ((a_team, seed), (1 - a_team, seed * 2 + 1)):
        net = CuarentaNet.from_checkpoint(checkpoints[team], value_target=targets[team])
        net.eval()
        nets[team] = net
        trackers[team] = HistoryTracker(env)
        searchers[team] = NetworkISMCTS(
            env, net, trackers[team],
            n_iter=cfg["n_iter"],
            max_history_len=cfg["max_history_len"],
            leaf_value_mode=cfg["leaf_value_mode"],
            max_rollout_len=cfg["max_rollout_len"],
            rng=random.Random(team_seed),
            dirichlet_eps=0.0,        # greedy evaluation: no exploration noise
            device=cfg["device"],
        )

    caidas = [0, 0]
    limpias = [0, 0]
    captures = [0, 0]
    carton = [0, 0]
    reward_sum = [0.0, 0.0]
    plies = 0
    started = time.perf_counter()

    while not env.state.done:
        p = int(env.state.current_player)
        team = TEAMS[p]
        legal = env.legal_actions()
        obs = env.observe(p)

        visits = searchers[team].search(root_player=p)
        a = (max(visits, key=visits.get) if sum(visits.values()) > 0
             else random.randrange(len(legal)))

        # Every tracker records every turn, not only its own side's: each search reads the
        # whole public history.
        for tracker in trackers.values():
            tracker.record(p, obs, a)

        # `legal[a]` IS the action the env applies, so counting the capture from it is
        # exact; the capture piles cannot be read back after a cartón clears them.
        played = legal[a]
        _, reward, _, _, info = env.step(a)

        reward_sum[team] += float(reward)
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
        plies += 1

    winner = env.state.winner
    if winner is None:
        raise RuntimeError("the game did not finish")

    for team in (0, 1):
        won = 1 if winner == team else 0
        expected = (captures[team] + 2 * caidas[team] + 2 * limpias[team]
                    + carton[team] + WIN_BONUS * won)
        if abs(reward_sum[team] - expected) > 1e-6:
            raise AssertionError(
                f"seed {seed}: team {team} accumulated return {reward_sum[team]} but "
                f"captures + 2*caidas + 2*limpias + carton + WIN_BONUS*won = {expected} "
                f"(captures={captures[team]}, caidas={caidas[team]}, "
                f"limpias={limpias[team]}, carton={carton[team]}, won={won})")

    b_team = 1 - a_team
    return {
        "seed": seed,
        "a_team": a_team,
        "b_team": b_team,
        "winner": winner,
        "a_won": 1 if winner == a_team else 0,
        "score_for": int(env.state.scores[a_team]),
        "score_against": int(env.state.scores[b_team]),
        "return_for": reward_sum[a_team],
        "return_against": reward_sum[b_team],
        "caidas_for": caidas[a_team],
        "caidas_against": caidas[b_team],
        "limpias_for": limpias[a_team],
        "limpias_against": limpias[b_team],
        "plies": plies,
        "seconds": time.perf_counter() - started,
    }


def worker(args) -> dict:
    try:
        return play_one(args)
    except Exception as exc:                  # keep the other games running
        seed, _ckpt_a, _ckpt_b, _cfg, a_team = args
        return {"seed": seed, "a_team": a_team,
                "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def win_interval(successes: int, n: int) -> Tuple[float, float, float, float]:
    """(rate, se, lo, hi) for a win rate, with the interval clamped to [0, 1]."""
    rate = successes / n if n else float("nan")
    se = float(np.sqrt(max(rate * (1.0 - rate), 1e-9) / n)) if n else float("nan")
    lo = max(0.0, rate - 1.96 * se)
    hi = min(1.0, rate + 1.96 * se)
    return rate, se, lo, hi


def report(rows: List[dict], setup: dict, elapsed: float,
           labels: Tuple[str, str]) -> None:
    ok = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]

    line = "=" * 72
    print(f"\n{line}")
    print(f"A: {labels[0]}   [{setup['value_target_a']}]")
    print(f"B: {labels[1]}   [{setup['value_target_b']}]")
    print(f"games: {len(ok)} played"
          + (f", {len(failed)} FAILED" if failed else "")
          + f"   |   {setup['n_iter']} sims/move, leaf value: {setup['leaf_value_mode']}")
    print(line)

    if not ok:
        print("no games completed")
        return

    wins = sum(r["a_won"] for r in ok)
    n = len(ok)
    rate, se, lo, hi = win_interval(wins, n)

    # With the team assignment alternating, a genuinely equal match is 50%: anything
    # inside ~2 standard errors of that is not a result.
    verdict = ("A is ahead" if lo > 0.5 else
               "B is ahead" if hi < 0.5 else
               "no clear winner (the interval covers 50%)")

    print(f"A win rate: {rate:6.1%}  ({wins}/{n}, ±{se:.1%} se, "
          f"95% CI {lo:.1%}..{hi:.1%})")
    print(f"B win rate: {1 - rate:6.1%}")
    print(f"  -> {verdict}")
    print()
    print(f"{'':22s}{'A':>10}{'B':>10}")
    for label, key in (("avg score", "score"), ("avg return", "return"),
                       ("avg caídas", "caidas"), ("avg limpias", "limpias")):
        print(f"  {label:20s}{mean([r[key + '_for'] for r in ok]):>10.2f}"
              f"{mean([r[key + '_against'] for r in ok]):>10.2f}")
    print(f"  {'avg plies':20s}{mean([r['plies'] for r in ok]):>10.1f}{'':>10}")
    print(f"  {'avg seconds/game':20s}{mean([r['seconds'] for r in ok]):>10.2f}{'':>10}")

    # Both seat assignments, over the same deals: a first-team advantage shows up here as
    # a gap that survives the pairing.
    print()
    for team in (0, 1):
        subset = [r for r in ok if r["a_team"] == team]
        if subset:
            print(f"  A on team {team} ({len(subset):3d} games): "
                  f"{mean([r['a_won'] for r in subset]):.1%} won")

    # Per deal, did the sides split it or did one win both games?
    by_seed: Dict[int, List[dict]] = {}
    for r in ok:
        by_seed.setdefault(r["seed"], []).append(r)
    pairs = [rs for rs in by_seed.values() if len(rs) == 2]
    if pairs:
        sweeps = sum(1 for rs in pairs if sum(x["a_won"] for x in rs) == 2)
        splits = sum(1 for rs in pairs if sum(x["a_won"] for x in rs) == 1)
        shutouts = sum(1 for rs in pairs if sum(x["a_won"] for x in rs) == 0)
        print(f"  paired deals ({len(pairs)}): A swept {sweeps}, split {splits}, "
              f"lost both {shutouts}")

    if failed:
        print()
        for r in failed[:5]:
            print(f"  seed {r['seed']} (A on team {r['a_team']}) failed: {r['error']}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    print(f"\ntotal time: {elapsed:.1f}s")
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_value_targets(spec: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse `--value-target`: 'auto' (both from their sidecars), one value for both, or a
    pair in checkpoint order. Each side's mode is independent, which is what lets a
    win-mode checkpoint play a return-mode one.
    """
    text = spec.strip().lower()
    if text in ("", "auto"):
        return None, None

    from network import VALUE_TARGETS

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2:
        raise SystemExit(f"--value-target wants 'auto', one value, or two values "
                         f"separated by a comma (got {spec!r})")
    for part in parts:
        if part not in VALUE_TARGETS:
            raise SystemExit(f"--value-target values must be {VALUE_TARGETS} (got {part!r})")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head evaluation: two ISMCTS checkpoints.")
    parser.add_argument("--checkpoint-a", type=str, required=True,
                        help="checkpoint A (the side every reported number is from)")
    parser.add_argument("--checkpoint-b", type=str, required=True,
                        help="checkpoint B (the opponent)")
    parser.add_argument("--games", type=int, default=20,
                        help="number of games (default 20); each deal is played twice, "
                             "once per team assignment, so keep this even")
    parser.add_argument("--n-iter", type=int, default=200,
                        help="ISMCTS simulations per move for BOTH sides (default 200, "
                             "the training value)")
    parser.add_argument("--max-history-len", type=int, default=32,
                        help="history window both searches read (default 32)")
    parser.add_argument("--leaf-value", type=str, default="net", choices=["net", "rollout"],
                        help="how both searches value their leaves (default: the net)")
    parser.add_argument("--max-rollout-len", type=int, default=512,
                        help="only used with --leaf-value rollout")
    parser.add_argument("--value-target", type=str, default="auto",
                        help="'auto' (default) reads each checkpoint's sidecar; one value "
                             "('win'/'return') applies to both; two values in checkpoint "
                             "order ('return,win') set them independently. The mode is not "
                             "in the weights, and the search plays a different agent "
                             "without it")
    parser.add_argument("--workers", type=int, default=0,
                        help="games played in parallel (0 = one per CPU)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch device for both searches (default cpu: batch-1 "
                             "forwards are faster there)")
    parser.add_argument("--seed-base", type=int, default=800_000,
                        help="deal i uses seed seed_base + i (default 800000, which does "
                             "not overlap the trainers' evaluation seeds)")
    parser.add_argument("--json", type=str, default=None,
                        help="write the per-game rows to this JSON file (with --sweep: one "
                             "entry per simulation budget)")
    parser.add_argument("--sweep", type=str, default=None,
                        help="comma-separated simulation budgets to run instead of a single "
                             "match, e.g. '32,128,400'. It separates 'the policies differ' "
                             "from 'the search differs', which is the first thing to check "
                             "when two runs land close. Ignores --n-iter.")
    args = parser.parse_args()

    for path, what in ((args.checkpoint_a, "A"), (args.checkpoint_b, "B")):
        if not Path(path).exists():
            raise SystemExit(f"checkpoint {what} not found: {path}")

    if args.sweep:
        try:
            budgets = [int(part) for part in args.sweep.split(",") if part.strip()]
        except ValueError:
            raise SystemExit(f"--sweep wants integers, got {args.sweep!r}")
        if any(budget < 1 for budget in budgets):
            raise SystemExit("--sweep budgets must be >= 1")
    else:
        budgets = [args.n_iter]

    value_target_a, value_target_b = resolve_value_targets(args.value_target)

    # What each side will ACTUALLY be searched in, resolved before the games rather than
    # left to `auto`: a silent fall back to 'win' on a return-mode checkpoint would look
    # like a result instead of a setting.
    from network import CuarentaNet

    resolved_a = CuarentaNet.resolve_value_target(args.checkpoint_a, value_target_a)
    resolved_b = CuarentaNet.resolve_value_target(args.checkpoint_b, value_target_b)

    setup = {
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "value_target_a": resolved_a,
        "value_target_b": resolved_b,
        "max_history_len": args.max_history_len,
        "leaf_value_mode": args.leaf_value,
        "max_rollout_len": args.max_rollout_len,
        "device": args.device,
    }

    # PAIRED DESIGN: each deal is played TWICE, once with A on team 0 and once on team 1.
    tasks = [{"seed": args.seed_base + i // 2, "a_team": i % 2}
             for i in range(args.games)]

    def run(condition_setup: dict) -> Tuple[List[dict], float]:
        payload = [(task["seed"], args.checkpoint_a, args.checkpoint_b,
                    condition_setup, task["a_team"]) for task in tasks]
        started = time.perf_counter()
        with Pool(processes=args.workers or None) as pool:
            rows = list(tqdm(pool.imap(worker, payload, chunksize=1),
                             total=len(payload), desc="games"))
        return rows, time.perf_counter() - started

    if len(budgets) == 1:
        condition_setup = dict(setup, n_iter=budgets[0])
        print(f"A: {args.checkpoint_a}  [{resolved_a}]")
        print(f"B: {args.checkpoint_b}  [{resolved_b}]")
        print(f"{len(tasks)} games, {condition_setup['n_iter']} sims/move")
        rows, elapsed = run(condition_setup)
        report(rows, condition_setup, elapsed, (args.checkpoint_a, args.checkpoint_b))
        if args.json:
            out = Path(args.json)
            out.write_text(json.dumps({"setup": condition_setup, "rows": rows}, indent=2))
            print(f"per-game rows -> {out}")
        return

    # ---- sweep over the search budget, same deals throughout ---------------
    print(f"Simulation-budget sweep: {len(tasks)} games per budget")
    print(f"A: {args.checkpoint_a}  [{resolved_a}]")
    print(f"B: {args.checkpoint_b}  [{resolved_b}]")
    print(f"budgets: {', '.join(str(b) for b in budgets)} sims/move\n")

    conditions = []
    for n_iter in budgets:
        condition_setup = dict(setup, n_iter=n_iter)
        print(f"--- {n_iter} simulation(s)/move ---")
        rows, elapsed = run(condition_setup)
        ok = [r for r in rows if "error" not in r]
        if ok:
            wins = sum(r["a_won"] for r in ok)
            rate, se, _lo, _hi = win_interval(wins, len(ok))
            print(f"  A {rate:.1%} ({wins}/{len(ok)}, ±{se:.1%} se), "
                  f"{mean([r['seconds'] for r in ok]):.2f}s/game")
        conditions.append({"setup": condition_setup, "rows": rows, "elapsed": elapsed})

    line = "=" * 72
    print(f"\n{line}")
    print("Checkpoint A vs B, by search budget (same deals throughout)")
    print(line)
    print(f"{'sims/move':>10}{'A win rate':>12}{'95% CI':>20}{'games':>8}{'verdict':>24}")
    for condition in conditions:
        ok = [r for r in condition["rows"] if "error" not in r]
        if not ok:
            print(f"{condition['setup']['n_iter']:>10}{'no games completed':>12}")
            continue
        wins = sum(r["a_won"] for r in ok)
        rate, _se, lo, hi = win_interval(wins, len(ok))
        verdict = ("A ahead" if lo > 0.5 else "B ahead" if hi < 0.5 else "too close to call")
        print(f"{condition['setup']['n_iter']:>10}{rate:>11.1%}"
              f"{f'{lo:.1%}..{hi:.1%}':>20}{len(ok):>8}{verdict:>24}")
    print(line)
    print("A gap that survives a rising budget is a difference in the POLICY; a gap that")
    print("closes is a difference in the search, or in what the search is fed.")
    print(line)

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(
            {"checkpoint_a": args.checkpoint_a, "checkpoint_b": args.checkpoint_b,
             "games": args.games, "seed_base": args.seed_base,
             "conditions": [{"n_iter": c["setup"]["n_iter"], "setup": c["setup"],
                             "elapsed": c["elapsed"], "rows": c["rows"]}
                            for c in conditions]},
            indent=2))
        print(f"per-game rows -> {out}")


if __name__ == "__main__":
    main()
