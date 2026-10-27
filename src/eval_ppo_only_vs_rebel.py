#!/usr/bin/env python3
"""
Evaluate the PPO policy (NO search) against the ReBeL agent, head to head.

WHAT THIS MEASURES
------------------
`src/ppo` and `src/rebel` are two independent pipelines that solve the same game by
different means: PPO reads the position and plays its policy head directly, while ReBeL
maintains a public belief state and runs a depth-limited CFR search over it at every
decision, using its network for the policy prior, the leaf values and the belief's
likelihood model. This script puts them in the same game.

WHY THE MATCH RUNS ACROSS TWO PROCESSES
---------------------------------------
Both packages are self-contained *script directories*, each with its own `env.py`,
`card.py`, `captures.py` and `utils.py`. That is deliberate (each pipeline can be read
and run on its own), but it also means one Python process cannot import both: `import
env` resolves to whichever directory reached `sys.path` first, and the second pipeline
would silently run on the FIRST one's modules. The failure would be quiet and the
numbers would look plausible.

So the match is split in two:

  * the ReBeL process is the REFEREE. It owns the single true game -- `src/rebel/core.py`
    deals, validates and applies every move -- plus the public belief state and the CFR
    solver, and it builds the observation and the candidate-action features for whoever
    is about to play. Because only one process knows the state, the two sides cannot
    drift apart or disagree about the rules.
  * the PPO process is a WORKER. It loads a PPO checkpoint and answers exactly one
    question: given (observation, action features, legality mask), which slot? It also
    owns the policy's recurrent memory, so the referee never has to touch a PPO tensor.

Driving PPO this way is legitimate because **all three pipelines consume the same
representation**: a 208-float observation plus 84 features per candidate action in at
most 64 slots, where slot `i` means "the i-th legal action". That layout is declared in
`src/ppo/utils.py: OBS_KEYS_ISMCTS`, `src/ismcts/history.py`, `src/rebel/core.py` and
consumed by the exported ONNX model and the Go backend. The worker re-validates the
shapes on every turn, so a layout break fails loudly instead of quietly weakening the
policy.

THE RECURRENCE CONTRACT (the part that is easy to get wrong)
-----------------------------------------------------------
The PPO network is recurrent, and it was trained with its memory advanced on EVERY turn
of the match -- including the three seats it was not controlling at that moment (see
`src/ppo/trainer.py`). So the referee sends every turn to the worker, the worker
advances its memory on all of them, and it returns an action only for PPO's own seats.
Advancing only on PPO's own turns would feed it a hidden state it never saw in training
and would quietly understate it -- the same bug that made PPO self-play train against a
handicapped opponent.

ReBeL needs the mirror-image treatment for its own reasons: `PublicBelief.refresh` and
`record_action` run on every turn, whoever played it, because the belief's Bayes update
scores the observed plays of ALL seats -- that is what keeps the particle set a valid
posterior over the hidden hands.

SEATS AND THE PAIRED DESIGN
---------------------------
PPO controls a TEAM (seats 0+2 or 1+3) and so does ReBeL. Every deal is then played
TWICE with the roles swapped, so the two games share a shuffle and only the side
controls differ. Cuarenta deals decide a lot, so same-deal pairing is the cheapest way
to see a real difference through the noise, and it turns the per-team split in the
report into a direct read on any first-team advantage. Metrics are reported from PPO's
point of view, matching the `_for` / `_against` convention of both pipelines' harnesses.

Both sides play greedily: PPO takes the argmax of its masked policy logits, and ReBeL
plays the CFR solver's root-probability argmax (no exploration noise).

REWARD IDENTITY SELF-CHECK
--------------------------
After every game the harness proves it counted what the env paid: each team's accumulated
reward must equal `captures + 2*caidas + 2*limpias + carton + WIN_BONUS*won` exactly. A
failed identity means the numbers in the report do not measure the game, so the game is
reported as failed rather than silently included.

Usage:
    python3 src/eval_ppo_only_vs_rebel.py --games 20
    python3 src/eval_ppo_only_vs_rebel.py --games 20 --n-iter 64 --depth 4
    python3 src/eval_ppo_only_vs_rebel.py --ppo-checkpoint checkpoints-ppo-return/best.pt \\
        --rebel-checkpoint checkpoints-rebel-return/best.pt --games 50 --json rebel.json

Timing: the ReBeL side is the cost -- every one of its decisions pays `n_iter` CFR
iterations over `--depth` plies of a `--particles`-particle belief -- so it is several
seconds per game. `--workers` (one match process per core) is the knob that matters for
anything above a few dozen games, and `--n-iter`/`--depth` are the ones to turn down
while exploring.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import Process, Queue
from multiprocessing.connection import Pipe
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PPO_DIR = REPO_ROOT / "src" / "ppo"
REBEL_DIR = REPO_ROOT / "src" / "rebel"


# ---------------------------------------------------------------------------
# PPO side: a worker process with PPO alone on its sys.path
# ---------------------------------------------------------------------------

def ppo_worker_main(conn, checkpoint: str, device: str) -> None:
    """
    Serve turn requests until the referee closes the pipe.

    Requests
      {"kind": "reset", "game": id}                      -> new memory for that game
      {"kind": "turn", ...}                              -> {"slot": int|None, "value": float}

    `slot` is None on turns the PPO seats do not control: the referee still sends those
    so the memory keeps spanning the whole match (see the module docstring).
    """
    sys.path.insert(0, str(PPO_DIR))

    import numpy as np
    import torch

    torch.set_num_threads(1)          # one core per worker; threading only adds jitter
    from networks import load_actor_critic

    net = load_actor_critic(checkpoint, torch.device(device))
    net.eval()

    memories: Dict[int, object] = {}

    while True:
        try:
            request = conn.recv()
        except EOFError:
            break
        if request is None:
            break

        game = request["game"]
        if request["kind"] == "reset":
            memories[game] = None
            conn.send({"ok": True})
            continue

        # The referee sends the flat observation (the layout every pipeline shares); the
        # worker only checks the size and shape, and fails loudly instead of feeding the
        # policy a vector it was not trained on.
        obs_flat = np.asarray(request["obs"], dtype=np.float32).reshape(-1)
        feats = np.asarray(request["feats"], dtype=np.float32)
        mask = np.asarray(request["mask"]) > 0.5
        n_legal = int(request["n_legal"])

        if obs_flat.shape[0] != net.obs_dim:
            raise ValueError(f"observation is {obs_flat.shape[0]} floats, "
                             f"the checkpoint wants {net.obs_dim}")
        if feats.shape != (net.n_slots, net.action_feat_dim):
            raise ValueError(f"action features are {feats.shape}, the checkpoint wants "
                             f"({net.n_slots}, {net.action_feat_dim})")
        if mask.shape[0] != net.n_slots:
            raise ValueError(f"mask is {mask.shape[0]} long, the checkpoint wants "
                             f"{net.n_slots}")
        if not (mask[:n_legal].all() and not mask[n_legal:].any()):
            raise ValueError("the mask must mark exactly the first n_legal slots; the "
                             "slot -> action mapping would be wrong otherwise")

        obs_t = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)
        feats_t = torch.as_tensor(feats).unsqueeze(0)
        mask_t = torch.as_tensor(mask).unsqueeze(0)

        with torch.no_grad():
            logits, value, new_memory = net(obs_t, feats_t, mask_t, memories.get(game))
        memories[game] = new_memory       # advanced on EVERY turn, acting or not

        slot: Optional[int] = None
        if request["act"]:
            legal_slots = mask_t[0].nonzero(as_tuple=True)[0]
            slot = int(legal_slots[int(torch.argmax(logits[0][legal_slots]).item())].item())

        conn.send({"slot": slot, "value": float(value.item())})


class PPOWorker:
    """Referee-side handle to the PPO worker process."""

    def __init__(self, checkpoint: str, device: str):
        self._conn, child_conn = Pipe()
        self._proc = Process(target=ppo_worker_main,
                             args=(child_conn, checkpoint, device))
        self._proc.start()
        self._games = 0

    def new_game(self) -> int:
        self._games += 1
        self._conn.send({"kind": "reset", "game": self._games})
        self._recv()
        return self._games

    def turn(self, game: int, obs, feats, mask, n_legal: int, act: bool) -> Optional[int]:
        """Feed one turn; returns PPO's slot when it is PPO's turn, else None."""
        self._conn.send({"kind": "turn", "game": game, "obs": obs, "feats": feats,
                         "mask": mask, "n_legal": n_legal, "act": act})
        return self._recv()["slot"]

    def _recv(self) -> dict:
        try:
            return self._conn.recv()
        except EOFError as exc:                      # the worker's traceback is on stderr
            raise RuntimeError("the PPO worker process died (see its traceback above)") from exc

    def close(self) -> None:
        try:
            self._conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(timeout=30)


# ---------------------------------------------------------------------------
# ReBeL side: the referee, which owns the game and the belief
# ---------------------------------------------------------------------------

def play_game(task: dict, setup: dict, worker: PPOWorker, ctx: dict) -> dict:
    """
    Play one game and return its row of statistics.

    `task` is {"game": i, "seed": s, "ppo_team": 0|1}; `ctx` holds the modules imported
    by the match process (they are looked up at call time, not module import time,
    because this file's top level must stay importable without either pipeline on the
    path -- a spawned child re-imports it).
    """
    CuarentaEnv = ctx["CuarentaEnv"]
    TEAMS = ctx["TEAMS"]
    WIN_BONUS = ctx["WIN_BONUS"]
    action_features_matrix = ctx["action_features_matrix"]
    flat_obs_for = ctx["flat_obs_for"]
    clone_state = ctx["clone_state"]
    PublicBelief = ctx["PublicBelief"]
    belief_policy_fn = ctx["belief_policy_fn"]
    choose_action = ctx["choose_action"]
    ReBeLSolver = ctx["ReBeLSolver"]
    has_reason = ctx["has_reason"]
    PointsReasons = ctx["PointsReasons"]

    seed = task["seed"]
    ppo_team = task["ppo_team"]
    rebel_team = 1 - ppo_team

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    rng = random.Random(seed)
    belief = PublicBelief(n_particles=setup["particles"], rng=rng,
                          history=setup["belief_history"])
    # The belief scores observed plays with the likelihood model of the net it represents,
    # so it is told which team the net plays for: PPO's plays are scored the same way, as
    # an opponent the posterior has to explain.
    belief.policy_fn = belief_policy_fn(setup["rebel_net"], belief,
                                        allowed_teams=(rebel_team,))
    solver = ReBeLSolver(net=setup["rebel_net"], n_iter=setup["n_iter"],
                         depth=setup["depth"], rng=rng, device=setup["device"])

    game = worker.new_game()
    caidas = [0, 0]
    limpias = [0, 0]
    captures = [0, 0]
    carton = [0, 0]
    reward_sum = [0.0, 0.0]
    plies = 0
    leaf_evals = 0
    started = time.perf_counter()

    while not env.state.done:
        state = env.state
        seat = int(state.current_player)
        legal = env.legal_actions()
        ppo_turn = (TEAMS[seat] == ppo_team)

        # The belief is refreshed for whoever is about to act, then updated with the play
        # once it is chosen -- every seat's play feeds it, which is what keeps the
        # particle set a posterior over the hidden hands.
        belief.refresh(state, seat)

        if ppo_turn:
            # Every turn goes to the worker, so PPO's memory spans the match exactly as it
            # did during training; on ReBeL's turns the answer is simply ignored.
            obs = flat_obs_for(seat, state)
            feats, mask = action_features_matrix(legal, state)
            slot = worker.turn(game, obs, feats, mask, len(legal), ppo_turn)
            pick = int(slot)
            # PPO is not one of our solvers, so nothing authorises a likelihood for this
            # play beyond the belief's own model.
            strategy = None
        else:
            result = solver.solve(state, seat, belief)
            leaf_evals += int(result.leaf_evals)
            pick = choose_action(result.root_probs, rng, explore=False)
            # The strategy table travels with the record so the belief can score this
            # play by how likely the search itself thought it was.
            strategy = result.strategy

        belief.record_action(clone_state(state), seat, legal[pick], strategy=strategy)

        # What the env is about to apply is exactly `legal[pick]`, which is where the
        # +1-per-captured-card term of the reward comes from (the capture piles cannot be
        # read back after the step: a cartón clears them).
        played = legal[pick]

        # step returns (observation, reward, done, truncated, info); the loop condition
        # reads env.state.done, so only the reward and info matter here.
        _obs, reward, _done, _truncated, info = env.step(pick)

        team = int(TEAMS[seat])
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
    ppo_won = int(winner == ppo_team)

    # Self-check of the whole harness: the env's rewards are dense (+1 per captured card,
    # +2 caída, +2 limpia, the cartón points, WIN_BONUS for the win -- see the REWARD note
    # in src/ismcts/env.py), so they must decompose EXACTLY as
    # `return == captures + 2*caidas + 2*limpias + carton + WIN_BONUS*won` for both teams.
    # If this ever fails, the numbers below are not measuring what they claim to.
    for team in (0, 1):
        won = 1 if winner == team else 0
        expected = (captures[team] + 2 * caidas[team] + 2 * limpias[team]
                    + carton[team] + WIN_BONUS * won)
        if abs(reward_sum[team] - expected) > 1e-6:
            raise AssertionError(
                f"game {game}: team {team} accumulated return {reward_sum[team]} but "
                f"captures + 2*caidas + 2*limpias + carton + WIN_BONUS*won = {expected} "
                f"(captures={captures[team]}, caidas={caidas[team]}, "
                f"limpias={limpias[team]}, carton={carton[team]}, won={won})")

    return {
        "game": task["game"],
        "seed": seed,
        "ppo_team": ppo_team,
        "rebel_team": rebel_team,
        "winner": winner,
        "ppo_won": ppo_won,
        "score_for": env.state.scores[ppo_team],
        "score_against": env.state.scores[rebel_team],
        "return_for": reward_sum[ppo_team],
        "return_against": reward_sum[rebel_team],
        "caidas_for": caidas[ppo_team],
        "caidas_against": caidas[rebel_team],
        "limpias_for": limpias[ppo_team],
        "limpias_against": limpias[rebel_team],
        "plies": plies,
        "leaf_evals": leaf_evals,
        "seconds": time.perf_counter() - started,
    }


def match_main(tasks: List[dict], result_queue: Queue, setup: dict) -> None:
    """Play a slice of the games: import ReBeL, start a PPO worker, loop."""
    sys.path.insert(0, str(REBEL_DIR))

    import torch

    torch.set_num_threads(1)
    from belief import DEFAULT_HISTORY, PublicBelief
    from core import (TEAMS, CuarentaEnv, action_features_matrix, clone_state,
                      flat_obs_for)
    from env import WIN_BONUS
    from network import load_rebel_checkpoint
    from selfplay import belief_policy_fn, choose_action
    from solver import ReBeLSolver
    from utils import PointsReasons, has_reason

    net = load_rebel_checkpoint(setup["rebel_checkpoint"], torch.device(setup["device"]))
    net.eval()

    ctx = {
        "CuarentaEnv": CuarentaEnv,
        "TEAMS": TEAMS,
        "WIN_BONUS": WIN_BONUS,
        "action_features_matrix": action_features_matrix,
        "flat_obs_for": flat_obs_for,
        "clone_state": clone_state,
        "PublicBelief": PublicBelief,
        "belief_policy_fn": belief_policy_fn,
        "choose_action": choose_action,
        "ReBeLSolver": ReBeLSolver,
        "PointsReasons": PointsReasons,
        "has_reason": has_reason,
    }
    local_setup = dict(setup, rebel_net=net,
                       belief_history=setup["belief_history"] or DEFAULT_HISTORY)

    worker = PPOWorker(setup["ppo_checkpoint"], setup["device"])
    try:
        for task in tasks:
            try:
                result_queue.put(play_game(task, local_setup, worker, ctx))
            except Exception as exc:                      # keep other games running
                result_queue.put({"game": task["game"], "seed": task["seed"],
                                  "error": f"{type(exc).__name__}: {exc}"})
    finally:
        worker.close()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stderr_of_rate(successes: int, n: int) -> float:
    """Binomial standard error of a win rate: sqrt(p(1-p)/n)."""
    if n == 0:
        return float("nan")
    p = successes / n
    return (p * (1 - p) / n) ** 0.5


def win_interval(successes: int, n: int):
    """(rate, se, lo, hi) for a win rate, with the interval clamped to [0, 1]."""
    rate = successes / n if n else float("nan")
    se = stderr_of_rate(successes, n)
    lo = max(0.0, rate - 1.96 * se)
    hi = min(1.0, rate + 1.96 * se)
    return rate, se, lo, hi


def report(rows: List[dict], setup: dict, elapsed: float, ppo_label: str,
           rebel_label: str) -> None:
    ok = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]

    line = "=" * 72
    print(f"\n{line}")
    print(f"PPO (no search): {ppo_label}")
    print(f"ReBeL:           {rebel_label}")
    print(f"games: {len(ok)} played"
          + (f", {len(failed)} FAILED" if failed else "")
          + f"   |   {setup['n_iter']} CFR iters/move, depth {setup['depth']}, "
            f"{setup['particles']} particles")
    print(line)

    if not ok:
        print("no games completed")
        return

    wins = sum(r["ppo_won"] for r in ok)
    n = len(ok)
    rate, se, lo, hi = win_interval(wins, n)

    # The bar: with the team assignment alternating, a genuinely equal match is 50%.
    # Anything inside ~2 standard errors of that is not a result.
    verdict = ("PPO is ahead" if lo > 0.5 else
               "ReBeL is ahead" if hi < 0.5 else
               "no clear winner (the interval covers 50%)")

    print(f"PPO win rate:      {rate:6.1%}  ({wins}/{n}, ±{se:.1%} se, "
          f"95% CI {lo:.1%}..{hi:.1%})")
    print(f"ReBeL win rate:    {1 - rate:6.1%}")
    print(f"  -> {verdict}")
    print()
    print(f"{'':22s}{'PPO':>10}{'ReBeL':>10}")
    for label, key in (("avg score", "score"), ("avg return", "return"),
                       ("avg caídas", "caidas"), ("avg limpias", "limpias")):
        print(f"  {label:20s}{mean([r[key + '_for'] for r in ok]):>10.2f}"
              f"{mean([r[key + '_against'] for r in ok]):>10.2f}")
    print(f"  {'avg plies':20s}{mean([r['plies'] for r in ok]):>10.1f}{'':>10}")
    print(f"  {'avg seconds/game':20s}{mean([r['seconds'] for r in ok]):>10.2f}{'':>10}")
    print(f"  {'avg leaf evals':20s}{mean([r['leaf_evals'] for r in ok]):>10.1f}{'':>10}")

    # Both side assignments, over the same deals. A real first-team advantage shows up
    # here as a gap that survives the pairing.
    print()
    for team in (0, 1):
        subset = [r for r in ok if r["ppo_team"] == team]
        if subset:
            print(f"  PPO on team {team} ({len(subset):3d} games): "
                  f"{mean([r['ppo_won'] for r in subset]):.1%} won")

    # Paired view: per deal, did the sides split it or did one win both games?
    by_seed: Dict[int, List[dict]] = {}
    for r in ok:
        by_seed.setdefault(r["seed"], []).append(r)
    pairs = [rs for rs in by_seed.values() if len(rs) == 2]
    if pairs:
        sweeps = sum(1 for rs in pairs if sum(x["ppo_won"] for x in rs) == 2)
        splits = sum(1 for rs in pairs if sum(x["ppo_won"] for x in rs) == 1)
        shutouts = sum(1 for rs in pairs if sum(x["ppo_won"] for x in rs) == 0)
        print(f"  paired deals ({len(pairs)}): PPO swept {sweeps}, split {splits}, "
              f"lost both {shutouts}")

    if failed:
        print()
        for r in failed[:5]:
            print(f"  game {r['game']} (seed {r['seed']}) failed: {r['error']}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    print(f"\ntotal time: {elapsed:.1f}s")
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_ppo_checkpoint() -> Path:
    """`best.pt` once a run has been accepted, otherwise the warm start."""
    best = REPO_ROOT / "checkpoints-ppo" / "best.pt"
    return best if best.exists() else REPO_ROOT / "checkpoints-ppo" / "bootstrap.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head evaluation: PPO (no search) vs ReBeL.")
    parser.add_argument("--ppo-checkpoint", type=str, default=str(default_ppo_checkpoint()),
                        help="PPO checkpoint (default: checkpoints-ppo/best.pt if it "
                             "exists, else bootstrap.pt)")
    parser.add_argument("--rebel-checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints-rebel" / "best.pt"),
                        help="ReBeL checkpoint (default: checkpoints-rebel/best.pt)")
    parser.add_argument("--games", type=int, default=20,
                        help="number of games (default 20); each deal is played twice, "
                             "once per team assignment, so keep this even")
    parser.add_argument("--n-iter", type=int, default=32,
                        help="CFR iterations per ReBeL decision (default 32, the "
                             "training value)")
    parser.add_argument("--depth", type=int, default=3,
                        help="plies of ReBeL search before the net values a leaf "
                             "(default 3)")
    parser.add_argument("--particles", type=int, default=16,
                        help="belief particles in ReBeL's chance node (default 16)")
    parser.add_argument("--belief-history", type=int, default=None,
                        help="recent actions the belief's Bayes update scores (default: "
                             "belief.DEFAULT_HISTORY)")
    parser.add_argument("--workers", type=int, default=1,
                        help="match processes to run in parallel; each one carries its "
                             "own PPO worker (default 1)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch device for both policies (default cpu: these are "
                             "batch-1 forwards, where CPU beats MPS)")
    parser.add_argument("--seed-base", type=int, default=800_000,
                        help="deal i uses seed seed_base + i (default 800000, which does "
                             "not overlap the trainers' evaluation seeds)")
    parser.add_argument("--json", type=str, default=None,
                        help="write the per-game rows to this JSON file (with --sweep: "
                             "one entry per CFR budget)")
    parser.add_argument("--sweep", type=str, default=None,
                        help="comma-separated CFR-iteration budgets to run instead of a "
                             "single match, e.g. '1,4,16,32'. The curve shows how much of "
                             "ReBeL's strength comes from the search rather than from the "
                             "network, though even at 1 iteration the strategy is belief-"
                             "weighted, so it is not a pure prior comparison. Ignores "
                             "--n-iter.")
    args = parser.parse_args()

    ppo_checkpoint = Path(args.ppo_checkpoint)
    rebel_checkpoint = Path(args.rebel_checkpoint)
    for path, what in ((ppo_checkpoint, "PPO checkpoint"),
                       (rebel_checkpoint, "ReBeL checkpoint")):
        if not path.exists():
            raise SystemExit(f"{what} not found: {path}")

    if args.sweep:
        try:
            budgets = [int(part) for part in args.sweep.split(",") if part.strip()]
        except ValueError:
            raise SystemExit(f"--sweep wants integers, got {args.sweep!r}")
        if any(b < 1 for b in budgets):
            raise SystemExit("--sweep budgets must be >= 1")
    else:
        budgets = [args.n_iter]

    setup = {
        "ppo_checkpoint": str(ppo_checkpoint.resolve()),
        "rebel_checkpoint": str(rebel_checkpoint.resolve()),
        "depth": args.depth,
        "particles": args.particles,
        "belief_history": args.belief_history,
        "device": args.device,
    }

    # PAIRED DESIGN: each deal is played TWICE, once with PPO on team 0 and once on team
    # 1. Same-deal pairing cancels most of the luck of the shuffle (the dominant source
    # of variance here, since a deal decides a lot in Cuarenta), and it makes the team
    # split below a direct read on any first-team advantage rather than on different
    # deals. The two games are not mirror images -- the policies differ -- so a pair does
    # not have to split one win each.
    tasks = [{"game": i, "seed": args.seed_base + i // 2, "ppo_team": i % 2}
             for i in range(args.games)]

    workers = max(1, min(args.workers, len(tasks)))
    # Contiguous chunks, not strided: `tasks[i::workers]` would hand each worker only ONE
    # of the two team assignments (the assignment alternates with the task index), which
    # would confound any per-process effect with the side being measured.
    per_worker = (len(tasks) + workers - 1) // workers
    chunks = [tasks[i:i + per_worker] for i in range(0, len(tasks), per_worker)]

    def collect(condition_setup: dict, quiet: bool) -> (List[dict], float):
        """Run every task against one ReBeL configuration and return its rows."""
        if not quiet:
            print(f"PPO (no search) vs ReBeL: {len(tasks)} games, {workers} worker(s)")
            print(f"  PPO:    {ppo_checkpoint}")
            print(f"  ReBeL:  {rebel_checkpoint}  "
                  f"({condition_setup['n_iter']} CFR iters/move, depth "
                  f"{condition_setup['depth']})")

        result_queue: Queue = Queue()
        processes = [Process(target=match_main, args=(chunk, result_queue, condition_setup))
                     for chunk in chunks if chunk]
        for proc in processes:
            proc.start()

        rows: List[dict] = []
        started = time.perf_counter()
        interactive = sys.stdout.isatty()
        tick = max(1, len(tasks) // 10)
        try:
            while len(rows) < len(tasks):
                rows.append(result_queue.get())
                done = len(rows)
                errors = sum(1 for r in rows if "error" in r)
                if quiet:
                    continue
                if interactive:
                    progress = f"\r  {done}/{len(tasks)} games"
                    if errors:
                        progress += f" ({errors} failed)"
                    print(progress, end="", flush=True)
                elif done % tick == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} games"
                          + (f" ({errors} failed)" if errors else ""), flush=True)
        except KeyboardInterrupt:
            print("\ninterrupted: terminating workers")
            for proc in processes:
                proc.terminate()
            raise SystemExit(130)
        finally:
            for proc in processes:
                proc.join(timeout=60)
        if not quiet:
            print()
        return rows, time.perf_counter() - started

    if len(budgets) == 1:
        n_iter = budgets[0]
        condition_setup = dict(setup, n_iter=n_iter)
        rows, elapsed = collect(condition_setup, quiet=False)
        report(rows, condition_setup, elapsed, str(ppo_checkpoint), str(rebel_checkpoint))
        if args.json:
            out = Path(args.json)
            out.write_text(json.dumps({"setup": condition_setup, "rows": rows}, indent=2))
            print(f"per-game rows -> {out}")
        return

    # ---- sweep: the same deals at several CFR budgets ----------------------
    print(f"CFR-budget sweep: {len(tasks)} games per budget, {workers} worker(s)")
    print(f"  PPO:    {ppo_checkpoint}")
    print(f"  ReBeL:  {rebel_checkpoint}")
    print(f"  budgets: {', '.join(str(b) for b in budgets)} CFR iters/move\n")

    conditions = []
    for n_iter in budgets:
        condition_setup = dict(setup, n_iter=n_iter)
        print(f"--- {n_iter} CFR iteration(s)/move ---")
        rows, elapsed = collect(condition_setup, quiet=False)
        conditions.append({"setup": condition_setup, "rows": rows, "elapsed": elapsed})

    line = "=" * 72
    print(f"\n{line}")
    print("PPO (no search) vs ReBeL, by ReBeL CFR budget (same deals throughout)")
    print(line)
    print(f"{'iters/move':>10}{'PPO win rate':>14}{'95% CI':>20}{'games':>8}{'verdict':>26}")
    for condition in conditions:
        ok = [r for r in condition["rows"] if "error" not in r]
        if not ok:
            print(f"{condition['setup']['n_iter']:>10}{'no games completed':>14}")
            continue
        wins = sum(r["ppo_won"] for r in ok)
        n = len(ok)
        rate, se, lo, hi = win_interval(wins, n)
        verdict = ("PPO ahead" if lo > 0.5 else "ReBeL ahead" if hi < 0.5 else "too close to call")
        print(f"{condition['setup']['n_iter']:>10}{rate:>13.1%}{f'{lo:.1%}..{hi:.1%}':>20}"
              f"{n:>8}{verdict:>26}")
    print(line)
    print("The curve is what the search adds on top of the ReBeL network: the belief and")
    print("the CFR strategy are still in play at low budgets, so even the first row is not")
    print("a bare policy comparison.")
    print(line)

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(
            {"ppo_checkpoint": str(ppo_checkpoint), "rebel_checkpoint": str(rebel_checkpoint),
             "games": args.games, "seed_base": args.seed_base,
             "conditions": [{"n_iter": c["setup"]["n_iter"], "setup": c["setup"],
                             "elapsed": c["elapsed"], "rows": c["rows"]}
                            for c in conditions]},
            indent=2))
        print(f"per-game rows -> {out}")


if __name__ == "__main__":
    main()
