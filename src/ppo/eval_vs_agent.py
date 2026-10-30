#!/usr/bin/env python3
"""
Evaluate ONE PPO checkpoint against a fixed bot: `--agent-type heuristic|random`.

WHAT THIS ANSWERS
=================
`PPOTrainer.evaluate` runs a five-match league every `--eval-every` steps so a training
run can see whether it is improving: it gates against the reference snapshot and reports
the heuristic and a random bot among the matches. That league is a training instrument —
`--eval-games` games (60 by default), a fixed seed base, and numbers that land in wandb
next to the losses.

This script is the other half: a one-off measurement of a FILE, against the opponent(s)
you name, with as many games as you are willing to pay for, and a report you can read
without a dashboard:

    python3 src/ppo/eval_vs_agent.py --checkpoint checkpoints-ppo/best_heuristic.pt

    # both bots on the SAME deals, so the two rates are directly comparable
    python3 src/ppo/eval_vs_agent.py --checkpoint checkpoints-ppo/bootstrap.pt \
        --agent-type heuristic random --games 400 --workers 8

    # two checkpoints on identical deals: run it twice with the same --seed-base
    python3 src/ppo/eval_vs_agent.py --checkpoint A.pt --seed-base 910_000
    python3 src/ppo/eval_vs_agent.py --checkpoint B.pt --seed-base 910_000

It does not re-implement the match: it calls `eval_harness.evaluate`, so the games, the
seat rotation and the metric names (`eval/<name>`, `eval/<name>_score_for`, `_stderr`, …)
are exactly the ones the trainer logs.

WHY 50% IS THE BAR, EVEN AGAINST A "WEAK" BOT
=============================================
The policy under test plays ONE seat, and the seat rotates with the game index
(0, 1, 2, 3, 0, …), so deals — not seats — decide the difference: an even match is 50% in
expectation however strong the opponent is. Which is why "50% vs random" is a terrible
result: random hands back captures it does not want. 50% vs the heuristic is the project's
yardstick — the raw served policy sits near 49% against it, the search-backed one near 77%
(see AGENTS.md).

What the expectation does NOT hand you is precision. 60 games is ±6% standard error, 200
is ±3.5%, and only ~1000 starts to resolve a 3-point gap. The report prints the standard
error and a 95% interval, and the verdict line refuses to call anything the interval does
not support. Keep `--games` a multiple of 4: the rotation then gives each seat exactly
`games / 4` matches.

THE SEEDS ARE FIXED, SO RUNS ARE PAIRED
=======================================
Game `i` uses seed `--seed-base + i`, and EVERY opponent in a run gets the same seeds, so
`--agent-type heuristic random` is a paired comparison: both bots face the same shuffles
and a difference between the two rates is not a difference between deals. The default base
(900_000) overlaps neither the trainer's evaluation seeds (500_000) nor the ISMCTS
head-to-head script's (800_000).

The policy plays greedily (argmax over the legal slots) and the bots are seeded per game,
so a (checkpoint, seed) pair reproduces its match exactly. The recurrent state is advanced
on EVERY turn — the seats the policy does not control included — because that is the
contract the network was trained under; `eval_harness` owns it, this script only reads the
results. Games run in worker processes on CPU (batch-1 forwards are faster there, and the
pool is what makes a few hundred games cheap); `--workers 0` means one per CPU.
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import DEFAULT_SAVE_DIR
from eval_harness import OpponentSpec, evaluate
from networks import load_actor_critic

# What a rate against each bot means, printed with its report so the number is not read
# in a vacuum. Only the two bots this script can put on the other side of the table.
YARDSTICKS = {
    "heuristic": (
        "the project's yardstick (rules bot: caída, limpia, biggest capture). The raw "
        "served policy sits near 49%; beating it is the first real result"),
    "random": (
        "sanity floor (uniform legal action): the policy should win almost every game, so "
        "a rate near 50% means the policy is not playing"),
}


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------

def win_interval(successes: int, n: int) -> Tuple[float, float, float, float]:
    """(rate, se, lo, hi) for a win rate; the interval is clamped to [0, 1]."""
    rate = successes / n if n else float("nan")
    se = float(np.sqrt(max(rate * (1.0 - rate), 1e-9) / n)) if n else float("nan")
    return rate, se, max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)


def verdict(lo: float, hi: float) -> str:
    """Read the interval against 50%, the fair bar for a seat-rotated match."""
    if lo > 0.5:
        return "ahead of the bot (the interval clears 50%)"
    if hi < 0.5:
        return "behind the bot (the interval clears 50% the other way)"
    return "no clear difference (the interval covers 50%)"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_opponent(kind: str, results: List[dict]) -> Dict[str, float]:
    """Print one opponent's report and return the summarised numbers."""
    n = len(results)
    wins = sum(r["won"] for r in results)
    rate, se, lo, hi = win_interval(wins, n)

    def m(key: str) -> float:
        return float(np.mean([r[key] for r in results]))

    print(f"\nvs {kind}   ({n} games, greedy policy)")
    print(f"  win rate        {rate:6.1%}   ({wins}/{n}, ±{se:.1%} se, "
          f"95% CI {lo:.1%}..{hi:.1%})")
    print(f"  verdict         {verdict(lo, hi)}")
    print(f"  score           for {m('score_for'):6.1f}   against {m('score_against'):6.1f}"
          f"   (game points)")
    print(f"  return          for {m('return_for'):6.1f}   against "
          f"{m('return_against'):6.1f}   (dense env reward)")
    print(f"  caídas          for {m('caidas_for'):6.2f}   against "
          f"{m('caidas_against'):6.2f}")
    print(f"  limpias         for {m('limpias_for'):6.2f}   against "
          f"{m('limpias_against'):6.2f}")
    print(f"  plies/game      {m('plies'):6.1f}")

    # The rotation should make these four numbers agree; a seat that stands out means the
    # rotation did not do its job (e.g. --games not a multiple of 4, or a different deck).
    seats = [f"seat {seat}: "
             f"{np.mean([r['won'] for r in results if r['policy_seat'] == seat]):.0%}"
             for seat in range(4) if any(r["policy_seat"] == seat for r in results)]
    if seats:
        print(f"  by seat         {', '.join(seats)}")
    print(textwrap.fill(f"what this means  {YARDSTICKS[kind]}", width=96,
                        initial_indent="  ", subsequent_indent=" " * 19))

    return {
        "games": n,
        "wins": wins,
        "win_rate": rate,
        "stderr": se,
        "ci_low": lo,
        "ci_high": hi,
        "score_for": m("score_for"),
        "score_against": m("score_against"),
        "return_for": m("return_for"),
        "return_against": m("return_against"),
        "caidas_for": m("caidas_for"),
        "caidas_against": m("caidas_against"),
        "limpias_for": m("limpias_for"),
        "limpias_against": m("limpias_against"),
        "plies": m("plies"),
    }


def report_summary(summary: Dict[str, Dict[str, float]], checkpoint: str,
                   arch: Dict[str, object], n_games: int, seed_base: int) -> None:
    """The one-table read of a multi-opponent run."""
    if len(summary) < 2:
        return
    line = "=" * 78
    print(f"\n{line}")
    print(f"{checkpoint}   [{'lstm' if arch.get('use_lstm') else 'feedforward'}, greedy]")
    print(f"{n_games} games per opponent, seeds {seed_base}..{seed_base + n_games - 1} "
          f"(same deals for every opponent)")
    print(line)
    print(f"{'opponent':10s}{'win rate':>10}{'±se':>8}{'95% CI':>18}   verdict")
    for kind, row in summary.items():
        ci = f"{row['ci_low']:.1%}..{row['ci_high']:.1%}"
        print(f"{kind:10s}{row['win_rate']:>10.1%}{row['stderr']:>8.1%}{ci:>18}   "
              f"{verdict(row['ci_low'], row['ci_high'])}")
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a PPO checkpoint against a heuristic or random bot.")
    p.add_argument("--checkpoint", "-c", type=str, required=True,
                   help="PPO checkpoint to evaluate, e.g. checkpoints-ppo/best_heuristic.pt "
                        "(the supervised warm start is checkpoints-ppo/bootstrap.pt)")
    p.add_argument("--agent-type", type=str, nargs="+", default=["heuristic"],
                   choices=sorted(YARDSTICKS),
                   help="which bot(s) the policy plays: 'heuristic' (the project's "
                        "yardstick, default) and/or 'random' (sanity floor). Both in one "
                        "run share the deals, so the rates are comparable")
    p.add_argument("--games", type=int, default=200,
                   help="games per opponent (default 200, about ±3.5%% standard error; "
                        "keep it a multiple of 4 so each seat gets games/4 matches)")
    p.add_argument("--workers", type=int, default=0,
                   help="games played in parallel (0 = one per CPU)")
    p.add_argument("--seed-base", type=int, default=900_000,
                   help="deal i uses seed seed_base + i (default 900000, which overlaps "
                        "neither the trainer's 500000 evaluation seeds nor the ISMCTS "
                        "head-to-head script's 800000). Reuse a base to compare two "
                        "checkpoints on identical deals")
    p.add_argument("--json", type=str, default=None,
                   help="also write the summary and the per-game rows to this JSON file")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}\n"
                         f"      checkpoints land in {DEFAULT_SAVE_DIR}/ and that "
                         f"directory is gitignored; the warm start is bootstrap.pt")
    if args.games < 1:
        parser.error("--games must be >= 1")
    if args.workers < 0:
        parser.error("--workers must be >= 0 (0 = one per CPU)")

    # De-duplicate while keeping the order the user typed: `--agent-type random random`
    # is one match, not two identical ones.
    agent_types = list(dict.fromkeys(args.agent_type))

    # Loaded on CPU on purpose: the worker processes play on CPU regardless (`_worker` in
    # eval_harness pins the device), so a GPU would only make this file look faster.
    try:
        policy = load_actor_critic(str(path), torch.device("cpu"))
    except (KeyError, RuntimeError) as exc:
        raise SystemExit(f"cannot evaluate {path}: {exc}") from exc

    arch = policy.architecture_config()
    print(f"checkpoint: {path}")
    print(f"architecture: {'lstm' if arch.get('use_lstm') else 'feedforward'}, "
          f"hidden {arch.get('hidden')}, obs {arch.get('obs_dim')}, "
          f"action features {arch.get('action_feat_dim')}")
    print(f"opponents: {', '.join(agent_types)}   |   games each: {args.games}   |   "
          f"seeds {args.seed_base}..{args.seed_base + args.games - 1}")
    if args.games % 4:
        print(f"note: --games {args.games} is not a multiple of 4, so the seat rotation "
              f"leaves some seats with one game fewer")

    summary: Dict[str, Dict[str, float]] = {}
    rows: Dict[str, List[dict]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for kind in agent_types:
        started = time.perf_counter()
        out = evaluate(
            policy,
            {kind: OpponentSpec(kind=kind)},
            n_games=args.games,
            seed_base=args.seed_base,
            workers=args.workers or None,
            keep_results_for=kind,
        )
        results = out.pop("_results")
        elapsed = time.perf_counter() - started
        metrics[kind] = out
        rows[kind] = results
        summary[kind] = report_opponent(kind, results)
        print(f"  ({elapsed:.1f}s, {elapsed / max(args.games, 1):.2f}s/game)")

    report_summary(summary, str(path), arch, args.games, args.seed_base)

    if args.json:
        out_path = Path(args.json)
        out_path.write_text(json.dumps(
            {
                "checkpoint": str(path),
                "architecture": arch,
                "games": args.games,
                "seed_base": args.seed_base,
                "workers": args.workers,
                "summary": summary,
                # The `eval/<name>_*` scalars, under the names the trainer logs, so a
                # result here can be dropped next to a wandb curve without renaming.
                "metrics": metrics,
                "rows": rows,
            },
            indent=2,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        ))
        print(f"summary + per-game rows -> {out_path}")


if __name__ == "__main__":
    main()
