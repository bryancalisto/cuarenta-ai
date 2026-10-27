#!/usr/bin/env python3
"""
Evaluate the ISMCTS agent against the ReBeL agent, head to head.

WHAT THIS MEASURES
------------------
`src/ismcts` and `src/rebel` are two independent pipelines that solve the same game by
different means: ISMCTS samples a fresh determinization at every decision and runs PUCT
over it, using its network for priors and leaf values; ReBeL maintains a public belief
state and runs depth-limited CFR over it, using its network for the policy prior, the
leaf values and the belief's likelihood model. This script puts them in the same game.

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
    solver, and it reports the result.
  * the ISMCTS process is a WORKER. It keeps a MIRROR of the same game in its own env,
    feeds every move into its `HistoryTracker` (its search reads the whole public
    history) and, on its own turns, answers with the most-visited root child of its
    search.

WHY THE WORKER MIRRORS THE GAME
-------------------------------
Unlike a policy-only player, a search needs a game state, and each pipeline's state
class is its own. So the referee sends the action it just applied -- by CONTENT, as
`(card index, capture card indices)`, never by slot number -- and the worker applies the
same move to its own env before searching. Both directions are content-checked: a move
the mirror env does not consider legal, or a final score the two envs disagree on,
raises instead of playing on. That is what makes the mirror safe to rely on: the two
ports are pinned by `src/ismcts/test_reward_parity.py` and friends, and this script
turns any drift into a loud failure rather than a quiet handicap.

TURN CONTRACT
-------------
Both sides are stateful between turns, for different reasons: ISMCTS's `HistoryTracker`
records every turn (whoever played it) because its search reads the public history, and
ReBeL's `PublicBelief.refresh` / `record_action` run on every turn because the Bayes
update scores the observed plays of ALL seats to keep the particles a valid posterior.
So each turn the referee refreshes its belief, applies its own move or asks the worker
for one, and records the move into the belief; the worker mirrors every move and answers
only for its own seats.

The referee's ask is always: "here is the move applied since the last message (or None
at the start); if it is my turn, give me yours." ISMCTS's answer is the argmax of its
root visit counts (`dirichlet_eps=0`, no exploration noise); ReBeL's is the argmax of
the CFR solver's root probabilities.

SEATS AND THE PAIRED DESIGN
---------------------------
ISMCTS controls a TEAM (seats 0+2 or 1+3) and so does ReBeL. Every deal is then played
TWICE with the roles swapped, so the two games share a shuffle and only the side
controls differ. Cuarenta deals decide a lot, so same-deal pairing is the cheapest way
to see a real difference through the noise, and it turns the per-team split in the
report into a direct read on any first-team advantage. Metrics are reported from
ISMCTS's point of view, matching the `_for` / `_against` convention of both pipelines'
harnesses.

REWARD IDENTITY SELF-CHECK
--------------------------
After every game the referee proves it counted what the env paid: each team's accumulated
reward must equal `captures + 2*caidas + 2*limpias + carton + WIN_BONUS*won` exactly. A
failed identity means the numbers in the report do not measure the game, so the game is
reported as failed rather than silently included.

Usage:
    python3 src/eval_ismcts_vs_rebel.py --games 20
    python3 src/eval_ismcts_vs_rebel.py --games 20 --ismcts-n-iter 400 --rebel-n-iter 64
    python3 src/eval_ismcts_vs_rebel.py --value-target return \\
        --ismcts-checkpoint checkpoints-return/best.pt \\
        --rebel-checkpoint checkpoints-rebel-return/best.pt --games 50 --json ismcts_rebel.json

Timing: BOTH sides search, so a game costs roughly what the two searches cost together
-- ISMCTS pays `--ismcts-n-iter` simulations per one of its turns, ReBeL pays
`--rebel-n-iter` CFR iterations over `--depth` plies per one of ITS turns. `--workers`
(one match process per core) is the knob that matters for anything above a few dozen
games.
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
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
ISMCTS_DIR = REPO_ROOT / "src" / "ismcts"
REBEL_DIR = REPO_ROOT / "src" / "rebel"

# An action as it crosses the process boundary: the played card's 0-39 index plus the
# 0-39 indices of the cards it captures (empty when it is a placement). Card indices are
# the same in all three pipelines (suit * 10 + rank position), so neither side has to
# trust the other's slot ordering.
ActionContent = Tuple[int, Tuple[int, ...]]


def action_content(action) -> ActionContent:
    """The portable form of one legal action."""
    capture = action.capture or ()
    return (int(action.card.index), tuple(sorted(int(c.index) for c in capture)))


def find_action_index(legal: Sequence, content: Optional[ActionContent]) -> Optional[int]:
    """Index of the action `content` describes in `legal`, or None if it is not there."""
    if content is None:
        return None
    for index, action in enumerate(legal):
        if action_content(action) == content:
            return index
    return None


def mirror_move(env, tracker, content: Optional[ActionContent]) -> None:
    """
    Apply the referee's last move to a mirror env, checking that it is legal there.

    A move the mirror does not consider legal means the two rule ports have drifted --
    fail loudly instead of quietly playing a different game.
    """
    if content is None:
        return
    legal = env.legal_actions()
    index = find_action_index(legal, content)
    if index is None:
        raise ValueError(f"the referee played {content}, which is not a legal action in "
                         f"the mirror env")
    seat = int(env.state.current_player)
    # The tracker wants the observation BEFORE the move and the action index, so its
    # search can replay this turn exactly as it happened.
    tracker.record(seat, env.observe(seat), index)
    env.step(index)


def isolate_pipeline(directory: Path, other: Path) -> None:
    """
    Put `directory` first on sys.path and drop any already-imported module that came from
    `other`.

    The two packages both ship `env.py`, `card.py`, `captures.py`, `utils.py` and
    `network.py`. Whichever directory is imported first must win, but a process started
    from its parent's state can already hold the other pipeline's modules under those
    names, and a bare `import env` would then silently use the wrong one -- the exact
    failure this whole two-process design exists to prevent.
    """
    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None) or ""
        if origin.startswith(str(other) + "/") or origin.startswith(str(other)):
            del sys.modules[name]
    sys.path.insert(0, str(directory))


# ---------------------------------------------------------------------------
# ISMCTS side: a worker process with ISMCTS alone on its sys.path
# ---------------------------------------------------------------------------

def ismcts_worker_main(conn, checkpoint: str, value_target: Optional[str], setup: dict) -> None:
    """
    Serve turn requests until the referee closes the pipe.

    Requests
      {"kind": "reset", "game": id, "seed": s}           -> {"ok": True}
      {"kind": "turn", "action": content|None, "act": b}  -> {"action": content|None}
      {"kind": "finish"}                                 -> {"scores", "winner"}

    Every message carries the move the referee applied since the previous one; the
    worker mirrors it, then answers only when the turn is its own (`act`).
    """
    isolate_pipeline(ISMCTS_DIR, REBEL_DIR)

    import torch

    torch.set_num_threads(1)          # one core per worker; threading only adds jitter
    from env import CuarentaEnv
    from history import HistoryTracker
    from network import CuarentaNet
    from network_ismcts_1 import NetworkISMCTS

    net = CuarentaNet.from_checkpoint(checkpoint, value_target=value_target)
    net.eval()

    games: Dict[int, dict] = {}

    def new_game(game: int, seed: int) -> dict:
        env = CuarentaEnv(seed=seed)
        env.reset(seed=seed)
        tracker = HistoryTracker(env)
        searcher = NetworkISMCTS(
            env, net, tracker,
            n_iter=setup["ismcts_n_iter"],
            max_history_len=setup["max_history_len"],
            leaf_value_mode=setup["leaf_value"],
            max_rollout_len=setup["max_rollout_len"],
            # Independent of the referee's RNG: the determinizations are this side's own
            # randomness, and the same seed makes the same game reproducible.
            rng=random.Random(seed * 2 + 1),
            dirichlet_eps=0.0,        # greedy evaluation: no exploration noise
            device=setup["device"],
        )
        games[game] = {"env": env, "tracker": tracker, "searcher": searcher}
        return games[game]

    while True:
        try:
            request = conn.recv()
        except EOFError:
            break
        if request is None:
            break

        if request["kind"] == "reset":
            new_game(request["game"], request["seed"])
            conn.send({"ok": True})
            continue

        state = games[request["game"]]

        if request["kind"] == "finish":
            env = state["env"]
            mirror_move(env, state["tracker"], request["action"])
            if env.state.winner is None:
                raise RuntimeError("the mirror env did not finish: the last move was "
                                   "never mirrored")
            conn.send({"scores": [int(s) for s in env.state.scores],
                       "winner": int(env.state.winner)})
            continue

        env = state["env"]
        tracker = state["tracker"]
        searcher = state["searcher"]

        mirror_move(env, tracker, request["action"])

        if not request["act"]:
            conn.send({"action": None})
            continue

        legal = env.legal_actions()
        seat = int(env.state.current_player)
        visits = searcher.search(root_player=seat)
        if sum(visits.values()) > 0:
            index = max(visits, key=visits.get)
        else:                                       # no visit at all: any legal move
            index = random.randrange(len(legal))
        conn.send({"action": action_content(legal[index])})


class ISMCTSWorker:
    """Referee-side handle to the ISMCTS worker process."""

    def __init__(self, checkpoint: str, value_target: Optional[str], setup: dict):
        self._conn, child_conn = Pipe()
        self._proc = Process(target=ismcts_worker_main,
                             args=(child_conn, checkpoint, value_target, setup))
        self._proc.start()
        self._games = 0

    def new_game(self, seed: int) -> int:
        self._games += 1
        self._conn.send({"kind": "reset", "game": self._games, "seed": seed})
        self._recv()
        return self._games

    def turn(self, game: int, content: Optional[ActionContent],
             act: bool) -> Optional[ActionContent]:
        """Mirror one move; returns ISMCTS's action content when it is its turn."""
        self._conn.send({"kind": "turn", "game": game, "action": content, "act": act})
        return self._recv()["action"]

    def finish(self, game: int, content: Optional[ActionContent]) -> dict:
        self._conn.send({"kind": "finish", "game": game, "action": content})
        return self._recv()

    def _recv(self) -> dict:
        try:
            return self._conn.recv()
        except EOFError as exc:                      # the worker's traceback is on stderr
            raise RuntimeError("the ISMCTS worker process died (see its traceback above)") from exc

    def close(self) -> None:
        try:
            self._conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(timeout=30)


# ---------------------------------------------------------------------------
# ReBeL side: the referee, which owns the game and the belief
# ---------------------------------------------------------------------------

def play_game(task: dict, setup: dict, worker: ISMCTSWorker, ctx: dict) -> dict:
    """
    Play one game and return its row of statistics.

    `task` is {"game": i, "seed": s, "ismcts_team": 0|1}; `ctx` holds the modules
    imported by the match process (they are looked up at call time, not module import
    time, because this file's top level must stay importable without either pipeline on
    the path -- a spawned child re-imports it).
    """
    CuarentaEnv = ctx["CuarentaEnv"]
    TEAMS = ctx["TEAMS"]
    WIN_BONUS = ctx["WIN_BONUS"]
    clone_state = ctx["clone_state"]
    PublicBelief = ctx["PublicBelief"]
    belief_policy_fn = ctx["belief_policy_fn"]
    choose_action = ctx["choose_action"]
    ReBeLSolver = ctx["ReBeLSolver"]
    has_reason = ctx["has_reason"]
    PointsReasons = ctx["PointsReasons"]

    seed = task["seed"]
    ismcts_team = task["ismcts_team"]
    rebel_team = 1 - ismcts_team

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    rng = random.Random(seed)
    belief = PublicBelief(n_particles=setup["particles"], rng=rng,
                          history=setup["belief_history"])
    # The belief scores observed plays with the likelihood model of the net it represents,
    # so it is told which team the net plays for: ISMCTS's plays are scored the same way,
    # as an opponent the posterior has to explain.
    belief.policy_fn = belief_policy_fn(setup["rebel_net"], belief,
                                        allowed_teams=(rebel_team,))
    solver = ReBeLSolver(net=setup["rebel_net"], n_iter=setup["rebel_n_iter"],
                         depth=setup["depth"], rng=rng, device=setup["device"])

    game = worker.new_game(seed)
    caidas = [0, 0]
    limpias = [0, 0]
    captures = [0, 0]
    carton = [0, 0]
    reward_sum = [0.0, 0.0]
    plies = 0
    leaf_evals = 0
    started = time.perf_counter()
    pending: Optional[ActionContent] = None       # the move not yet mirrored

    def step_env(seat: int, legal, pick: int, strategy) -> None:
        """Record into the belief, apply the move, and count every reward term."""
        nonlocal plies, pending
        belief.record_action(clone_state(env.state), seat, legal[pick], strategy=strategy)
        played = legal[pick]
        pending = action_content(played)
        _obs, reward, _done, _truncated, info = env.step(pick)
        team = int(TEAMS[seat])
        reward_sum[team] += float(reward)
        if played.capture:
            captures[team] += len(played.capture)
        reasons = info.get("reasons") or []
        # caída and limpia are mutually exclusive in the env's own reward (if/elif), so
        # mirror that here to keep the identity below exact.
        if has_reason(reasons, PointsReasons.CAIDA):
            caidas[team] += 1
        elif has_reason(reasons, PointsReasons.LIMPIA):
            limpias[team] += 1
        # The cartón is paid to the acting team on the move that ends a deal.
        carton[team] += int(info.get("carton_points") or 0)
        plies += 1

    while not env.state.done:
        state = env.state
        seat = int(state.current_player)
        legal = env.legal_actions()

        # ReBeL's search reads the belief for whoever is about to act, so it is
        # refreshed on every turn, not only its own.
        belief.refresh(state, seat)

        if TEAMS[seat] == rebel_team:
            result = solver.solve(state, seat, belief)
            leaf_evals += int(result.leaf_evals)
            pick = choose_action(result.root_probs, rng, explore=False)
            # The worker mirrors the previous move; nothing to answer for this one.
            worker.turn(game, pending, act=False)
            # The strategy table travels with the record so the belief can score this
            # play by how likely the search itself thought it was.
            step_env(seat, legal, pick, strategy=result.strategy)
        else:
            content = worker.turn(game, pending, act=True)
            index = find_action_index(legal, content)
            if index is None:
                raise ValueError(f"ISMCTS played {content}, which is not a legal action "
                                 f"in the referee env")
            # ISMCTS is not one of our solvers, so nothing authorises a likelihood for
            # this play beyond the belief's own model.
            step_env(seat, legal, index, strategy=None)

    # End-of-game parity check: the worker replayed the whole game in its own env, so its
    # final scores must be the referee's. A mismatch would mean the mirror was not the
    # same game, which would make the head-to-head meaningless.
    mirrored = worker.finish(game, pending)
    if mirrored["scores"] != [int(s) for s in env.state.scores] or \
            int(mirrored["winner"]) != int(env.state.winner):
        raise AssertionError(
            f"game {game}: the mirror env ended {mirrored}, the referee ended "
            f"{[int(s) for s in env.state.scores]} winner {int(env.state.winner)}")

    winner = env.state.winner
    if winner is None:
        raise RuntimeError("the referee env did not finish")
    ismcts_won = int(winner == ismcts_team)

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
        "ismcts_team": ismcts_team,
        "rebel_team": rebel_team,
        "winner": winner,
        "ismcts_won": ismcts_won,
        "score_for": int(env.state.scores[ismcts_team]),
        "score_against": int(env.state.scores[rebel_team]),
        "return_for": reward_sum[ismcts_team],
        "return_against": reward_sum[rebel_team],
        "caidas_for": caidas[ismcts_team],
        "caidas_against": caidas[rebel_team],
        "limpias_for": limpias[ismcts_team],
        "limpias_against": limpias[rebel_team],
        "plies": plies,
        "leaf_evals": leaf_evals,
        "seconds": time.perf_counter() - started,
    }


def match_main(tasks: List[dict], result_queue: Queue, setup: dict) -> None:
    """Play a slice of the games: import ReBeL, start an ISMCTS worker, loop."""
    isolate_pipeline(REBEL_DIR, ISMCTS_DIR)

    import torch

    torch.set_num_threads(1)
    from belief import DEFAULT_HISTORY, PublicBelief
    from core import TEAMS, CuarentaEnv, clone_state
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

    worker = ISMCTSWorker(setup["ismcts_checkpoint"], setup["value_target"], local_setup)
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


def report(rows: List[dict], setup: dict, elapsed: float, ismcts_label: str,
           rebel_label: str) -> None:
    ok = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]

    line = "=" * 72
    print(f"\n{line}")
    print(f"ISMCTS: {ismcts_label}")
    print(f"ReBeL:  {rebel_label}")
    print(f"games: {len(ok)} played"
          + (f", {len(failed)} FAILED" if failed else "")
          + f"   |   {setup['ismcts_n_iter']} sims/move vs "
            f"{setup['rebel_n_iter']} CFR iters/move (depth {setup['depth']}, "
            f"{setup['particles']} particles)")
    print(line)

    if not ok:
        print("no games completed")
        return

    wins = sum(r["ismcts_won"] for r in ok)
    n = len(ok)
    rate, se, lo, hi = win_interval(wins, n)

    # The bar: with the team assignment alternating, a genuinely equal match is 50%.
    # Anything inside ~2 standard errors of that is not a result.
    verdict = ("ISMCTS is ahead" if lo > 0.5 else
               "ReBeL is ahead" if hi < 0.5 else
               "no clear winner (the interval covers 50%)")

    print(f"ISMCTS win rate:   {rate:6.1%}  ({wins}/{n}, ±{se:.1%} se, "
          f"95% CI {lo:.1%}..{hi:.1%})")
    print(f"ReBeL win rate:    {1 - rate:6.1%}")
    print(f"  -> {verdict}")
    print()
    print(f"{'':22s}{'ISMCTS':>10}{'ReBeL':>10}")
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
        subset = [r for r in ok if r["ismcts_team"] == team]
        if subset:
            print(f"  ISMCTS on team {team} ({len(subset):3d} games): "
                  f"{mean([r['ismcts_won'] for r in subset]):.1%} won")

    # Paired view: per deal, did the sides split it or did one win both games?
    by_seed: Dict[int, List[dict]] = {}
    for r in ok:
        by_seed.setdefault(r["seed"], []).append(r)
    pairs = [rs for rs in by_seed.values() if len(rs) == 2]
    if pairs:
        sweeps = sum(1 for rs in pairs if sum(x["ismcts_won"] for x in rs) == 2)
        splits = sum(1 for rs in pairs if sum(x["ismcts_won"] for x in rs) == 1)
        shutouts = sum(1 for rs in pairs if sum(x["ismcts_won"] for x in rs) == 0)
        print(f"  paired deals ({len(pairs)}): ISMCTS swept {sweeps}, split {splits}, "
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head evaluation: ISMCTS vs ReBeL.")
    parser.add_argument("--ismcts-checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "best.pt"),
                        help="CuarentaNet checkpoint for ISMCTS (default: "
                             "checkpoints/best.pt)")
    parser.add_argument("--rebel-checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints-rebel" / "best.pt"),
                        help="ReBeL checkpoint (default: checkpoints-rebel/best.pt)")
    parser.add_argument("--games", type=int, default=20,
                        help="number of games (default 20); each deal is played twice, "
                             "once per team assignment, so keep this even")
    parser.add_argument("--ismcts-n-iter", type=int, default=200,
                        help="ISMCTS simulations per move (default 200, the training "
                             "value)")
    parser.add_argument("--max-history-len", type=int, default=32,
                        help="ISMCTS history window (default 32)")
    parser.add_argument("--value-target", type=str, default=None,
                        choices=["win", "return"],
                        help="value target the ISMCTS checkpoint was trained with; "
                             "defaults to the sidecar beside it, else win. It changes "
                             "what the ISMCTS search optimizes, so the wrong one is a "
                             "different opponent (see CuarentaNet.resolve_value_target)")
    parser.add_argument("--leaf-value", type=str, default="net", choices=["net", "rollout"],
                        help="how ISMCTS values its leaves (default: the net's value head)")
    parser.add_argument("--max-rollout-len", type=int, default=512,
                        help="only used with --leaf-value rollout")
    parser.add_argument("--rebel-n-iter", type=int, default=32,
                        help="CFR iterations per ReBeL decision (default 32, the "
                             "training value)")
    parser.add_argument("--depth", type=int, default=3,
                        help="plies of ReBeL search before the net values a leaf "
                             "(default 3)")
    parser.add_argument("--particles", type=int, default=16,
                        help="belief particles in ReBeL's chance node (default 16)")
    parser.add_argument("--belief-history", type=int, default=None,
                        help="recent actions ReBeL's belief Bayes update scores "
                             "(default: belief.DEFAULT_HISTORY)")
    parser.add_argument("--workers", type=int, default=1,
                        help="match processes to run in parallel; each one carries its "
                             "own ISMCTS worker (default 1)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch device for both networks (default cpu: these are "
                             "batch-1 forwards, where CPU beats MPS)")
    parser.add_argument("--seed-base", type=int, default=800_000,
                        help="deal i uses seed seed_base + i (default 800000, which does "
                             "not overlap the trainers' evaluation seeds)")
    parser.add_argument("--json", type=str, default=None,
                        help="write the per-game rows to this JSON file (with --sweep: "
                             "one entry per simulation budget)")
    parser.add_argument("--sweep", type=str, default=None,
                        help="comma-separated ISMCTS simulation budgets to run instead of "
                             "a single match, e.g. '1,8,32,128'. The curve shows how much "
                             "of ISMCTS's strength comes from search: at 1 simulation the "
                             "search is a no-op and you are comparing its POLICY against "
                             "ReBeL's search. Ignores --ismcts-n-iter.")
    args = parser.parse_args()

    ismcts_checkpoint = Path(args.ismcts_checkpoint)
    rebel_checkpoint = Path(args.rebel_checkpoint)
    for path, what in ((ismcts_checkpoint, "ISMCTS checkpoint"),
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
        budgets = [args.ismcts_n_iter]

    setup = {
        "ismcts_checkpoint": str(ismcts_checkpoint.resolve()),
        "rebel_checkpoint": str(rebel_checkpoint.resolve()),
        "value_target": args.value_target,
        "max_history_len": args.max_history_len,
        "leaf_value": args.leaf_value,
        "max_rollout_len": args.max_rollout_len,
        "rebel_n_iter": args.rebel_n_iter,
        "depth": args.depth,
        "particles": args.particles,
        "belief_history": args.belief_history,
        "device": args.device,
    }

    # PAIRED DESIGN: each deal is played TWICE, once with ISMCTS on team 0 and once on
    # team 1. Same-deal pairing cancels most of the luck of the shuffle (the dominant
    # source of variance here, since a deal decides a lot in Cuarenta), and it makes the
    # team split below a direct read on any first-team advantage rather than on different
    # deals. The two games are not mirror images -- the searchers differ -- so a pair does
    # not have to split one win each.
    tasks = [{"game": i, "seed": args.seed_base + i // 2, "ismcts_team": i % 2}
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
            print(f"ISMCTS vs ReBeL: {len(tasks)} games, {workers} worker(s)")
            print(f"  ISMCTS: {ismcts_checkpoint}  "
                  f"({condition_setup['ismcts_n_iter']} sims/move)")
            print(f"  ReBeL:  {rebel_checkpoint}  "
                  f"({condition_setup['rebel_n_iter']} CFR iters/move, depth "
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
        condition_setup = dict(setup, ismcts_n_iter=n_iter)
        rows, elapsed = collect(condition_setup, quiet=False)
        report(rows, condition_setup, elapsed, str(ismcts_checkpoint), str(rebel_checkpoint))
        if args.json:
            out = Path(args.json)
            out.write_text(json.dumps({"setup": condition_setup, "rows": rows}, indent=2))
            print(f"per-game rows -> {out}")
        return

    # ---- sweep: the same deals at several ISMCTS budgets -------------------
    print(f"ISMCTS-budget sweep: {len(tasks)} games per budget, {workers} worker(s)")
    print(f"  ISMCTS: {ismcts_checkpoint}")
    print(f"  ReBeL:  {rebel_checkpoint}")
    print(f"  budgets: {', '.join(str(b) for b in budgets)} sims/move\n")

    conditions = []
    for n_iter in budgets:
        condition_setup = dict(setup, ismcts_n_iter=n_iter)
        print(f"--- {n_iter} simulation(s)/move ---")
        rows, elapsed = collect(condition_setup, quiet=False)
        conditions.append({"setup": condition_setup, "rows": rows, "elapsed": elapsed})

    line = "=" * 72
    print(f"\n{line}")
    print("ISMCTS vs ReBeL, by ISMCTS search budget (same deals throughout)")
    print(line)
    print(f"{'sims/move':>10}{'ISMCTS win rate':>16}{'95% CI':>20}{'games':>8}{'verdict':>26}")
    for condition in conditions:
        ok = [r for r in condition["rows"] if "error" not in r]
        if not ok:
            print(f"{condition['setup']['ismcts_n_iter']:>10}{'no games completed':>16}")
            continue
        wins = sum(r["ismcts_won"] for r in ok)
        n = len(ok)
        rate, se, lo, hi = win_interval(wins, n)
        verdict = ("ISMCTS ahead" if lo > 0.5 else "ReBeL ahead" if hi < 0.5
                   else "too close to call")
        print(f"{condition['setup']['ismcts_n_iter']:>10}{rate:>15.1%}"
              f"{f'{lo:.1%}..{hi:.1%}':>20}{n:>8}{verdict:>26}")
    print(line)
    print("At 1 simulation the search is a no-op, so that row compares ISMCTS's POLICY")
    print("against ReBeL's search; every other row is what ISMCTS search adds on top.")
    print(line)

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(
            {"ismcts_checkpoint": str(ismcts_checkpoint),
             "rebel_checkpoint": str(rebel_checkpoint),
             "games": args.games, "seed_base": args.seed_base,
             "conditions": [{"ismcts_n_iter": c["setup"]["ismcts_n_iter"],
                             "setup": c["setup"], "elapsed": c["elapsed"],
                             "rows": c["rows"]}
                            for c in conditions]},
            indent=2))
        print(f"per-game rows -> {out}")


if __name__ == "__main__":
    main()
