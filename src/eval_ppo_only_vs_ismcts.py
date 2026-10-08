#!/usr/bin/env python3
"""
Evaluate the PPO policy (NO search) against the ISMCTS policy, head to head.

WHAT THIS MEASURES
------------------
`src/ppo` and `src/ismcts` are two independent pipelines that solve the same game by
different means: PPO reads the position and plays its policy head directly, while
ISMCTS runs a search from a fresh determinization at every decision and uses its
network only for priors and leaf values. This script puts them in the same game.

WHY THE MATCH RUNS ACROSS TWO PROCESSES
---------------------------------------
Both packages are self-contained *script directories*, each with its own `env.py`,
`card.py`, `captures.py` and `utils.py`. That is deliberate (each pipeline can be read
and run on its own), but it also means one Python process cannot import both: `import
env` resolves to whichever directory reached `sys.path` first, and the second pipeline
would silently run on the FIRST one's modules. The failure would be quiet and the
numbers would look plausible.

So the match is split in two:

  * the ISMCTS process is the REFEREE. It owns the single true game -- `src/ismcts/env.py`
    deals, validates and applies every move -- and it builds the observation and the
    candidate-action features for whoever is about to play. Because only one process
    knows the state, the two sides cannot drift apart or disagree about the rules.
  * the PPO process is a WORKER. It loads a PPO checkpoint and answers exactly one
    question: given (observation, action features, legality mask), which slot? It also
    owns the policy's recurrent memory, so the referee never has to touch a PPO tensor.

Driving PPO this way is legitimate because **both pipelines consume the same
representation**, declared twice on purpose:

  * observation: 208 floats, `src/ppo/utils.py: OBS_KEYS_ISMCTS` and
    `src/ismcts/history.py: _OBS_KEYS_BITVEC/_SCALAR/_VEC4` list the same keys in the
    same order (this is also what the exported ONNX model and the Go backend consume --
    see `backend/models/ppo_bootstrap.json`);
  * actions: 84 features per candidate action, at most 64 slots, and slot `i` means
    "the i-th legal action" (`action_features_matrix` exists in both packages).

The worker re-validates that on every turn (`check_obs_layout`, plus shape and mask
checks), so a layout break fails loudly instead of quietly weakening the policy.

THE RECURRENCE CONTRACT (the part that is easy to get wrong)
-----------------------------------------------------------
The PPO network is recurrent, and it was trained with its memory advanced on EVERY turn
of the match -- including the three seats it was not controlling at that moment (see
`src/ppo/trainer.py`). So the referee sends every turn to the worker, the worker
advances its memory on all of them, and it returns an action only for PPO's own seats.
Advancing only on PPO's own turns would feed it a hidden state it never saw in training
and would quietly understate it -- the same bug that made PPO self-play train against a
handicapped opponent.

ISMCTS needs the mirror-image treatment for its own reasons: `HistoryTracker.record` is
called for every turn, whoever played it, because its search reads the whole public
history.

SEATS AND THE PAIRED DESIGN
---------------------------
PPO controls a TEAM (seats 0+2 or 1+3) and so does ISMCTS. Every deal is then played
TWICE with the roles swapped, so the two games share a shuffle and only the side
controls differ. Cuarenta deals decide a lot, so same-deal pairing is the cheapest way
to see a real difference through the noise, and it turns the per-team split in the
report into a direct read on any first-team advantage. Metrics are reported from PPO's
point of view, matching the `_for` / `_against` convention of both pipelines' harnesses.

Both sides play greedily: PPO takes the argmax of its masked policy logits, and ISMCTS
plays the most-visited root child (`dirichlet_eps=0`, i.e. no exploration noise).

Usage:
    python3 src/eval_ppo_only_vs_ismcts.py --games 50
    python3 src/eval_ppo_only_vs_ismcts.py --games 50 --n-iter 400 --workers 4
    python3 src/eval_ppo_only_vs_ismcts.py --ppo-checkpoint checkpoints-ppo/best.pt \\
        --ismcts-checkpoint checkpoints/best.pt --games 100 --json results.json

Timing: the ISMCTS side is the cost, roughly `n_iter` per one of its two seats. Budget
a few seconds per game at the default 200 simulations, so `--workers` (one match
process per core) is the knob that matters for anything above ~50 games.
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
ISMCTS_DIR = REPO_ROOT / "src" / "ismcts"


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
    from utils import check_obs_layout, flatten_obs

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

        obs = request["obs"]
        feats = np.asarray(request["feats"], dtype=np.float32)
        mask = np.asarray(request["mask"]) > 0.5
        n_legal = int(request["n_legal"])

        # Fail loudly rather than hand the policy a vector it was not trained on.
        check_obs_layout(obs)
        obs_flat = flatten_obs(obs)
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
# ISMCTS side: the referee, which owns the game
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
    action_features_matrix = ctx["action_features_matrix"]
    HistoryTracker = ctx["HistoryTracker"]
    NetworkISMCTS = ctx["NetworkISMCTS"]
    has_reason = ctx["has_reason"]
    PointsReasons = ctx["PointsReasons"]

    seed = task["seed"]
    ppo_team = task["ppo_team"]
    ismcts_team = 1 - ppo_team

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    tracker = HistoryTracker(env)
    searcher = NetworkISMCTS(
        env, setup["ismcts_net"], tracker,
        n_iter=setup["n_iter"],
        max_history_len=setup["max_history_len"],
        leaf_value_mode=setup["leaf_value"],
        max_rollout_len=setup["max_rollout_len"],
        rng=random.Random(seed * 2 + 1),
        dirichlet_eps=0.0,               # greedy evaluation: no exploration noise
        device=setup["device"],
    )

    game = worker.new_game()
    caidas = [0, 0]
    limpias = [0, 0]
    reward_sum = [0.0, 0.0]
    plies = 0
    started = time.perf_counter()

    while not env.state.done:
        player = env.state.current_player
        legal = env.legal_actions()
        obs = env.observe(player)
        feats, mask = action_features_matrix(legal, env.state)
        ppo_turn = (TEAMS[player] == ppo_team)

        # Every turn goes to the worker, so PPO's memory spans the match exactly as it
        # did during training; on ISMCTS turns the answer is simply ignored.
        slot = worker.turn(game, obs, feats, mask, len(legal), ppo_turn)

        if ppo_turn:
            action_index = int(slot)
        else:
            visits = searcher.search(root_player=player)
            action_index = (max(visits, key=visits.get) if sum(visits.values()) > 0
                            else random.randrange(len(legal)))

        # ISMCTS reads the whole public history, so it records every turn, not just its own.
        tracker.record(player, obs, action_index)

        # step returns (observation, reward, done, truncated, info); the loop condition
        # reads env.state.done, so only the reward and info matter here.
        _obs, reward, _done, _truncated, info = env.step(action_index)

        team = info["team"]
        reward_sum[team] += float(reward)
        # caída and limpia are mutually exclusive in the env's own reward (if/elif), so
        # mirror that here to keep the identity below exact.
        if has_reason(info["reasons"], PointsReasons.CAIDA):
            caidas[team] += 1
        elif has_reason(info["reasons"], PointsReasons.LIMPIA):
            limpias[team] += 1
        plies += 1

    winner = env.state.winner
    ppo_won = int(winner == ppo_team)

    # Self-check of the whole harness: the env's rewards must decompose exactly as
    # return == 2*caidas + 2*limpias + 50*won, for both teams. If this ever fails, the
    # numbers below are not measuring what they claim to.
    for team in (0, 1):
        won = 1 if winner == team else 0
        expected = 2 * caidas[team] + 2 * limpias[team] + 50 * won
        if abs(reward_sum[team] - expected) > 1e-6:
            raise AssertionError(
                f"game {game}: team {team} accumulated return {reward_sum[team]} but "
                f"2*caidas + 2*limpias + 50*won = {expected} "
                f"(caidas={caidas[team]}, limpias={limpias[team]}, won={won})")

    return {
        "game": task["game"],
        "seed": seed,
        "ppo_team": ppo_team,
        "ismcts_team": ismcts_team,
        "winner": winner,
        "ppo_won": ppo_won,
        "score_for": env.state.scores[ppo_team],
        "score_against": env.state.scores[ismcts_team],
        "return_for": reward_sum[ppo_team],
        "return_against": reward_sum[ismcts_team],
        "caidas_for": caidas[ppo_team],
        "caidas_against": caidas[ismcts_team],
        "limpias_for": limpias[ppo_team],
        "limpias_against": limpias[ismcts_team],
        "plies": plies,
        "seconds": time.perf_counter() - started,
    }


def match_main(tasks: List[dict], result_queue: Queue, setup: dict) -> None:
    """Play a slice of the games: import ISMCTS, start a PPO worker, loop."""
    sys.path.insert(0, str(ISMCTS_DIR))

    import torch

    torch.set_num_threads(1)
    from env import CuarentaEnv, TEAMS, action_features_matrix
    from history import HistoryTracker
    from network import CuarentaNet
    from network_ismcts_1 import NetworkISMCTS
    from utils import PointsReasons, has_reason

    # The ISMCTS network is loaded once per process: it is an eval-mode nn.Module the
    # search only reads, and the per-decision state lives in the searcher, not the net.
    net = CuarentaNet()
    CuarentaNet.load_checkpoint(net, setup["ismcts_checkpoint"])
    net.eval()

    ctx = {
        "CuarentaEnv": CuarentaEnv,
        "TEAMS": TEAMS,
        "action_features_matrix": action_features_matrix,
        "HistoryTracker": HistoryTracker,
        "NetworkISMCTS": NetworkISMCTS,
        "PointsReasons": PointsReasons,
        "has_reason": has_reason,
    }
    local_setup = dict(setup, ismcts_net=net)

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
           ismcts_label: str) -> None:
    ok = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]

    line = "=" * 72
    print(f"\n{line}")
    print(f"PPO (no search): {ppo_label}")
    print(f"ISMCTS:          {ismcts_label}")
    print(f"games: {len(ok)} played"
          + (f", {len(failed)} FAILED" if failed else "")
          + f"   |   {setup['n_iter']} simulations/move   leaf value: {setup['leaf_value']}")
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
               "ISMCTS is ahead" if hi < 0.5 else
               "no clear winner (the interval covers 50%)")

    print(f"PPO win rate:      {rate:6.1%}  ({wins}/{n}, ±{se:.1%} se, "
          f"95% CI {lo:.1%}..{hi:.1%})")
    print(f"ISMCTS win rate:   {1 - rate:6.1%}")
    print(f"  -> {verdict}")
    print()
    print(f"{'':22s}{'PPO':>10}{'ISMCTS':>10}")
    for label, key in (("avg score", "score"), ("avg return", "return"),
                       ("avg caídas", "caidas"), ("avg limpias", "limpias")):
        print(f"  {label:20s}{mean([r[key + '_for'] for r in ok]):>10.2f}"
              f"{mean([r[key + '_against'] for r in ok]):>10.2f}")
    print(f"  {'avg plies':20s}{mean([r['plies'] for r in ok]):>10.1f}{'':>10}")
    print(f"  {'avg seconds/game':20s}{mean([r['seconds'] for r in ok]):>10.2f}{'':>10}")

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
        description="Head-to-head evaluation: PPO (no search) vs ISMCTS.")
    parser.add_argument("--ppo-checkpoint", type=str, default=str(default_ppo_checkpoint()),
                        help="PPO checkpoint (default: checkpoints-ppo/best.pt if it "
                             "exists, else bootstrap.pt)")
    parser.add_argument("--ismcts-checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "best.pt"),
                        help="CuarentaNet checkpoint for ISMCTS (default: checkpoints/best.pt)")
    parser.add_argument("--games", type=int, default=50,
                        help="number of games (default 50); each deal is played twice, "
                             "once per team assignment, so keep this even")
    parser.add_argument("--n-iter", type=int, default=200,
                        help="ISMCTS simulations per move (default 200, same as training)")
    parser.add_argument("--max-history-len", type=int, default=32,
                        help="ISMCTS history window (default 32)")
    parser.add_argument("--leaf-value", type=str, default="net", choices=["net", "rollout"],
                        help="how ISMCTS values its leaves (default: the net's value head)")
    parser.add_argument("--max-rollout-len", type=int, default=512,
                        help="only used with --leaf-value rollout")
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
                             "one entry per simulation budget)")
    parser.add_argument("--sweep", type=str, default=None,
                        help="comma-separated ISMCTS simulation budgets to run instead of "
                             "a single match, e.g. '1,8,32,128'. The resulting curve is "
                             "the quickest way to see how much of ISMCTS's strength comes "
                             "from search rather than from its network: at 1 simulation "
                             "the search is a no-op and you are comparing the two "
                             "POLICIES. Ignores --n-iter.")
    args = parser.parse_args()

    ppo_checkpoint = Path(args.ppo_checkpoint)
    ismcts_checkpoint = Path(args.ismcts_checkpoint)
    for path, what in ((ppo_checkpoint, "PPO checkpoint"), (ismcts_checkpoint, "ISMCTS checkpoint")):
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
        "ismcts_checkpoint": str(ismcts_checkpoint.resolve()),
        "max_history_len": args.max_history_len,
        "leaf_value": args.leaf_value,
        "max_rollout_len": args.max_rollout_len,
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
        """Run every task against one ISMCTS configuration and return its rows."""
        if not quiet:
            print(f"PPO (no search) vs ISMCTS: {len(tasks)} games, {workers} worker(s)")
            print(f"  PPO:    {ppo_checkpoint}")
            print(f"  ISMCTS: {ismcts_checkpoint}  "
                  f"({condition_setup['n_iter']} sims/move)")

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
        report(rows, condition_setup, elapsed, str(ppo_checkpoint), str(ismcts_checkpoint))
        if args.json:
            out = Path(args.json)
            out.write_text(json.dumps({"setup": condition_setup, "rows": rows}, indent=2))
            print(f"per-game rows -> {out}")
        return

    # ---- sweep: the same deals at several search budgets -------------------
    print(f"Search-budget sweep: {len(tasks)} games per budget, {workers} worker(s)")
    print(f"  PPO:    {ppo_checkpoint}")
    print(f"  ISMCTS: {ismcts_checkpoint}")
    print(f"  budgets: {', '.join(str(b) for b in budgets)} simulations/move\n")

    conditions = []
    for n_iter in budgets:
        condition_setup = dict(setup, n_iter=n_iter)
        print(f"--- {n_iter} simulation(s)/move ---")
        rows, elapsed = collect(condition_setup, quiet=False)
        conditions.append({"setup": condition_setup, "rows": rows, "elapsed": elapsed})

    line = "=" * 72
    print(f"\n{line}")
    print("PPO (no search) vs ISMCTS, by ISMCTS search budget (same deals throughout)")
    print(line)
    print(f"{'sims/move':>10}{'PPO win rate':>14}{'95% CI':>20}{'games':>8}{'verdict':>26}")
    for condition in conditions:
        ok = [r for r in condition["rows"] if "error" not in r]
        if not ok:
            print(f"{condition['setup']['n_iter']:>10}{'no games completed':>14}")
            continue
        wins = sum(r["ppo_won"] for r in ok)
        n = len(ok)
        rate, se, lo, hi = win_interval(wins, n)
        verdict = ("PPO ahead" if lo > 0.5 else "ISMCTS ahead" if hi < 0.5 else "too close to call")
        print(f"{condition['setup']['n_iter']:>10}{rate:>13.1%}{f'{lo:.1%}..{hi:.1%}':>20}"
              f"{n:>8}{verdict:>26}")
    print(line)
    print("At 1 simulation the search is a no-op, so that row compares the two POLICIES;")
    print("every additional row is what search adds on top of the ISMCTS network.")
    print(line)

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(
            {"ppo_checkpoint": str(ppo_checkpoint), "ismcts_checkpoint": str(ismcts_checkpoint),
             "games": args.games, "seed_base": args.seed_base,
             "conditions": [{"n_iter": c["setup"]["n_iter"], "setup": c["setup"],
                             "elapsed": c["elapsed"], "rows": c["rows"]}
                            for c in conditions]},
            indent=2))
        print(f"per-game rows -> {out}")


if __name__ == "__main__":
    main()
