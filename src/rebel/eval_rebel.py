"""
Evaluation for the ReBeL pipeline: the search + network's TEAM against a named LEAGUE.

WHAT IS BEING MEASURED
----------------------
Every member of the league is a policy that plays the team OPPOSITE to the one the
"network + search" under test controls. The opponent list is the one the PPO and ISMCTS
pipelines use, and every member is logged under the exact key family
`src/ppo/eval_harness.py` defines -- `eval/<member>` (win rate of the net's team),
`eval/<member>_stderr`, `_score_for/_against`, `_return_for/_against`,
`_caidas_for/_against`, `_limpias_for/_against`, `_plies`, `_n_games` -- so the three
pipelines' curves can be put on one chart. The names are BYTE-IDENTICAL on purpose: a
different spelling is a different chart.

League members (`LEAGUE_MEMBERS`, and what each one is):

    reference        the gated best-so-far checkpoint, `checkpoints-rebel/best.pt`
    heuristic        the rule-based HeuristicAgent (the stable external reference)
    initial          the frozen checkpoint the run started from (`--init-checkpoint`)
    prev_checkpoint  the most recently saved `policy_<iteration>.pt` before this evaluation
    random           a uniform-random legal-move policy

A member whose checkpoint does not exist yet (nothing saved, no best yet, no
`--init-checkpoint`) is SKIPPED for that evaluation: its keys simply do not appear. That
is the honest outcome -- a stale or zero number would look like a measurement of something
that was never played.

WHICH TEAM THE NET PLAYS, AND WHY IT ALTERNATES
----------------------------------------------
The two teams are seats {0, 2} and {1, 3}. The net plays team 0 on even games and team 1
on odd ones, so the match is not silently measuring a seat or first-player advantage: any
bias from who moves first is averaged out over the match instead of being absorbed into
the win rate. `_for` in every metric name always means the NET's team, whichever it was.

THE BELIEF DURING EVALUATION
----------------------------
The net's decisions use the same plumbing as self-play: refresh the belief for the acting
seat, search, play the argmax of the root strategy, then record the action with the
search's strategy table. Every opponent's plays are recorded with `strategy=None`, and the
belief's `policy_fn` is restricted to the net's own team (see `selfplay.belief_policy_fn`),
so an opponent play is treated as uninformative beyond its hard constraints -- the
heuristic, the random policy and a checkpoint member do NOT play the policy under test, and
pretending they do would bias the belief in a direction nothing supports.

WHAT EACH OPPONENT KIND PLAYS, AND WHY
--------------------------------------
`heuristic` and `random` are rule-based by definition. A `checkpoint` member plays the
ARGMAX OF ITS OWN POLICY (one network forward pass per ply), which is what
`src/ppo/eval_harness.py` -- the canonical contract -- does with a checkpoint opponent, and
it keeps a league evaluation's cost the same order as the heuristic match.

The alternative would be to let a checkpoint member run the same search the net under test
runs, which is what `src/ismcts/train.py` does for its `vs_reference`/`initial`/
`prev_checkpoint` members ("same search, same rules, only the weights differ"). That is the
stricter mirror, and it was not chosen here for two reasons: it doubles the search cost of
every league game (the member would solve at every ply, so a 30-game league match costs
twice what it does now), and it is NOT the PPO contract this family is being aligned with.
The consequence to keep in mind when reading `eval/reference`: this compares search+net
against the raw policy of another checkpoint, so an untrained value head (whose search is
little better than a random choice among the legal moves) can lose to its own raw policy --
measured: 0/8 in a mirror at n_iter=8/depth=2 with an untrained net. The mirrored numbers
become meaningful as the search gets stronger than the policy it improves.

Determinism: the env, the belief, the search and every opponent all derive from the game's
seed, and every player takes its argmax, so the same (checkpoint, member, seed) always
produces the same game -- which is what makes successive evaluations paired comparisons.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from belief import PublicBelief                         # noqa: E402
from env import WIN_BONUS
from core import (TEAMS, CuarentaEnv, action_features_matrix,  # noqa: E402
                  clone_state)
from heuristic_agent import HeuristicAgent              # noqa: E402
from selfplay import (SelfPlayConfig, belief_policy_fn,  # noqa: E402
                      build_input, choose_action)
from solver import ReBeLSolver                          # noqa: E402
from utils import PointsReasons, has_reason             # noqa: E402

DEFAULT_SAVE_DIR = "checkpoints-rebel"

# The league, in the order the summaries are printed. The first two are the "primary"
# members: they are the ones the gating and the long-standing reference use, and they run
# `--eval-games` games; the other three are the league extension and run `--league-games`.
LEAGUE_MEMBERS: Tuple[str, ...] = (
    "reference",
    "heuristic",
    "initial",
    "prev_checkpoint",
    "random",
)
PRIMARY_MEMBERS: Tuple[str, ...] = ("reference", "heuristic")
LEAGUE_ONLY_MEMBERS: Tuple[str, ...] = ("initial", "prev_checkpoint", "random")

# The canonical suffix family, WITHOUT the "eval/<member>" prefix. `match_metrics` emits
# exactly these and nothing else; the trainer's wandb families cover them all because every
# key starts with "eval/".
METRIC_SUFFIXES: Tuple[str, ...] = (
    "",
    "_stderr",
    "_score_for",
    "_score_against",
    "_return_for",
    "_return_against",
    "_caidas_for",
    "_caidas_against",
    "_limpias_for",
    "_limpias_against",
    "_plies",
    "_n_games",
)

_POLICY_FILE = re.compile(r"policy_(\d+)\.pt\Z")


# ---------------------------------------------------------------------------
# The league: who the net plays
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Opponent:
    """
    One league member: the name it is logged under, and the policy that plays the other
    team.

    `kind` is "heuristic" (the rule-based bot), "random" (uniform over the legal moves) or
    "checkpoint" (a ReBeL checkpoint saved by `save_rebel_checkpoint`, loaded in the worker
    through `network.load_rebel_checkpoint`, which re-reads the architecture from the file
    so a member with different hidden sizes still loads correctly).
    """
    name: str
    kind: str = "heuristic"
    checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in ("heuristic", "random", "checkpoint"):
            raise ValueError(f"unknown opponent kind: {self.kind!r}")
        if self.kind == "checkpoint" and not self.checkpoint:
            raise ValueError(f"member {self.name!r} is a checkpoint opponent with no path")


def latest_policy_checkpoint(save_dir, before: Optional[int] = None) -> Optional[Path]:
    """
    The most recently saved `policy_<iteration>.pt` in `save_dir`.

    `before` restricts the search to iterations STRICTLY below it -- the snapshots that
    already existed when the evaluation of iteration `before` runs (an iteration's own
    `policy_<n>.pt` is written after its evaluation, so it must not be picked). `before=None`
    means "the latest there is", which is what the standalone CLI wants.
    """
    best: Optional[Tuple[int, Path]] = None
    for path in Path(save_dir).glob("policy_*.pt"):
        match = _POLICY_FILE.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if before is not None and step >= before:
            continue
        if best is None or step > best[0]:
            best = (step, path)
    return None if best is None else best[1]


def league_members(*, save_dir=DEFAULT_SAVE_DIR, init_checkpoint: Optional[str] = None,
                   iteration: Optional[int] = None, include_league: bool = True
                   ) -> Tuple[Dict[str, Opponent], Dict[str, str]]:
    """
    The members that CAN be evaluated right now, plus the reason each absent one is absent.

    Returns `(members, skipped)`, both insertion-ordered to match `LEAGUE_MEMBERS`.
    `members` never contains a member whose checkpoint is missing: the caller logs exactly
    what is in it, so an unevaluable member produces NO keys rather than a stale number.

    `include_league=False` (the `--no-league` flag) drops the three extension members
    (`initial`, `prev_checkpoint`, `random`) and keeps `reference` and `heuristic`.
    """
    members: Dict[str, Opponent] = {}
    skipped: Dict[str, str] = {}

    reference = Path(save_dir) / "best.pt"
    if reference.exists():
        members["reference"] = Opponent("reference", "checkpoint", str(reference))
    else:
        skipped["reference"] = f"no {reference} yet (nothing gated as best so far)"

    members["heuristic"] = Opponent("heuristic", "heuristic")

    if not include_league:
        for name in LEAGUE_ONLY_MEMBERS:
            skipped[name] = "league disabled (--no-league)"
        return members, skipped

    if init_checkpoint is None:
        skipped["initial"] = "no --init-checkpoint"
    elif not Path(init_checkpoint).exists():
        skipped["initial"] = f"init checkpoint not found: {init_checkpoint}"
    else:
        members["initial"] = Opponent("initial", "checkpoint", str(init_checkpoint))

    previous = latest_policy_checkpoint(save_dir, before=iteration)
    if previous is None:
        skipped["prev_checkpoint"] = (
            f"no policy_<iteration>.pt below iteration {iteration} in {save_dir}"
            if iteration is not None else f"no policy_*.pt in {save_dir}")
    else:
        members["prev_checkpoint"] = Opponent("prev_checkpoint", "checkpoint", str(previous))

    members["random"] = Opponent("random", "random")
    return members, skipped


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

def _build_opponent(opponent: Opponent, seed: int, cfg: SelfPlayConfig):
    """
    What one opponent needs to act: a rule-based agent, or a loaded network.

    Exactly one of the two returned values is non-None. The checkpoint path is loaded here
    (per game, in whichever process plays it) rather than shipped as a state_dict, because
    the file already carries its own architecture -- that keeps the pool payload small.
    """
    if opponent.kind == "heuristic":
        return HeuristicAgent(seed=seed), None
    if opponent.kind == "checkpoint":
        import torch

        from network import load_rebel_checkpoint

        return None, load_rebel_checkpoint(opponent.checkpoint, torch.device(cfg.device))
    return None, None                    # "random" needs nothing but its rng


def _opponent_pick(opponent: Opponent, agent, opp_net, rng: random.Random, env, state,
                   seat: int, legal, belief: PublicBelief) -> int:
    """
    The opponent's slot index for this ply.

    All three kinds are deterministic given the game's seed: the heuristic and the random
    policy draw from their own `random.Random(seed * 2 + 1)`, and a checkpoint member plays
    the argmax of its policy over the same belief-conditioned input the search uses
    (`build_input`), i.e. it sees the position exactly the way this pipeline represents it.
    """
    if opponent.kind == "heuristic":
        return legal.index(agent.act(env, seat))
    if opponent.kind == "random":
        return rng.randrange(len(legal))
    obs = build_input(seat, state, belief)
    feats, _mask = action_features_matrix(legal, state)
    probs = opp_net.policy_probs(obs, feats, len(legal))
    return int(np.argmax(probs))


def play_eval_game(net, seed: int, cfg: SelfPlayConfig, net_team: int,
                   opponent: Optional[Opponent] = None) -> Dict:
    """
    Play one game: `net_team` (seats 0/2 or 1/3) is the search + network, the other team is
    `opponent` (the heuristic bot by default). Returns one game's summary; `*_for` fields
    are the NET's.

    Deterministic given `seed` and `opponent`: the env, the belief and the opponent all
    derive from the seed, and the net plays the search's argmax, so re-running the same
    evaluation reproduces the same games -- which is what makes successive evaluations
    comparable.
    """
    opponent = Opponent("heuristic", "heuristic") if opponent is None else opponent

    rng = random.Random(seed)

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    belief = PublicBelief(n_particles=cfg.n_particles, rng=rng, history=cfg.belief_history)
    belief.policy_fn = belief_policy_fn(net, belief, allowed_teams=(net_team,))
    solver = ReBeLSolver(net=net, n_iter=cfg.n_iter, depth=cfg.depth, rng=rng,
                         device=cfg.device)
    agent, opp_net = _build_opponent(opponent, seed, cfg)
    # The opponent's own rng, so its randomness is independent of the belief's particle
    # draws (sharing one generator would couple the two in a way nothing intends).
    opp_rng = random.Random(seed * 2 + 1)

    returns = [0.0, 0.0]
    caidas = [0, 0]
    limpias = [0, 0]
    cards = [0, 0]          # cards captured, the env's dense reward term
    carton = [0, 0]         # cartón points, paid to the acting team at the end of a deal
    plies = 0
    net_leaf_evals = 0

    while not env.state.done:
        state = env.state
        seat = int(state.current_player)
        legal = env.legal_actions()

        belief.refresh(state, seat)

        if TEAMS[seat] == net_team:
            result = solver.solve(state, seat, belief)
            net_leaf_evals += int(result.leaf_evals)
            pick = choose_action(result.root_probs, rng, explore=False)
            # The strategy table travels with the record so the belief can score this
            # play by how likely the search itself thought it was.
            strategy = result.strategy
        else:
            pick = _opponent_pick(opponent, agent, opp_net, opp_rng, env, state, seat,
                                  legal, belief)
            # Not one of our solvers: nothing authorises a likelihood for this play.
            strategy = None

        belief.record_action(clone_state(state), seat, legal[pick], strategy=strategy)

        # `legal[pick]` IS the action the env applies (it validates exactly this card/capture
        # pair before playing it), so counting the capture from it is exact. The previous
        # version instead diffed `cards_captured`, which silently undercounts the capture of
        # the last ply of a deal: a cartón payment resets the piles, so the delta goes
        # negative and that ply's captured cards were never counted.
        played = legal[pick]
        _obs, reward, _done, _truncated, info = env.step(pick)

        team = int(TEAMS[seat])
        returns[team] += float(reward)
        if played.capture:
            cards[team] += len(played.capture)
        reasons = info.get("reasons") or []
        # Caída and limpia are mutually exclusive, and the env's own reward tests caída
        # first; the same order here keeps this count equal to the reward decomposition.
        if has_reason(reasons, PointsReasons.CAIDA):
            caidas[team] += 1
        elif has_reason(reasons, PointsReasons.LIMPIA):
            limpias[team] += 1
        carton[team] += int(info.get("carton_points") or 0)
        plies += 1

    # Reward identity, per team. The env's reward is DENSE and has FIVE terms:
    #     +1 per captured card, +2 caída, +2 limpia, the cartón points, +WIN_BONUS for the win
    # so a team's return is `cards + 2*caidas + 2*limpias + carton + WIN_BONUS*won` (see the
    # REWARD note in src/ismcts/env.py, which all three pipelines follow).
    #
    # This assertion is the guard that THIS evaluator counts what it thinks it counts. It
    # now checks the identity EXACTLY, which is possible because the captures are counted
    # from the action that was played rather than from the env's capture piles (those are
    # reset by every cartón, which is what previously forced a weak `return >= accounted`).
    for team in (0, 1):
        won = 1.0 if env.state.winner == team else 0.0
        expected = (float(cards[team]) + 2.0 * caidas[team] + 2.0 * limpias[team]
                    + float(carton[team]) + WIN_BONUS * won)
        assert abs(returns[team] - expected) < 1e-9, (
            f"seed {seed} team {team}: return {returns[team]} != {expected} "
            f"(cards {cards[team]}, caidas {caidas[team]}, limpias {limpias[team]}, "
            f"carton {carton[team]}, won {won})"
        )

    opp_team = 1 - net_team
    return {
        "seed": seed,
        "opponent": opponent.name,
        "net_team": net_team,
        "winner": env.state.winner,
        "won": 1 if env.state.winner == net_team else 0,
        "score_for": int(env.state.scores[net_team]),
        "score_against": int(env.state.scores[opp_team]),
        "return_for": returns[net_team],
        "return_against": returns[opp_team],
        "caidas_for": caidas[net_team],
        "caidas_against": caidas[opp_team],
        "limpias_for": limpias[net_team],
        "limpias_against": limpias[opp_team],
        "plies": plies,
        "leaf_evals": net_leaf_evals,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def match_metrics(results: Sequence[Dict], prefix: str = "heuristic") -> Dict[str, float]:
    """
    Average game summaries into flat scalars, with the repository's metric names.

    The keys are the canonical family (`METRIC_SUFFIXES`) prefixed with `eval/<prefix>`:
    a win rate without its `n` is not interpretable, so `_stderr` and `_n_games` are always
    emitted alongside it. Empty input returns `{}` -- no keys at all -- which is how a
    skipped member stays invisible.
    """
    if not results:
        return {}
    n = len(results)

    def m(key: str) -> float:
        return float(np.mean([r[key] for r in results]))

    win_rate = m("won")
    # sqrt(p(1-p)/n): with 8 games the standard error is ~18%, which is why an evaluation
    # is only worth reading as a trend over several of them.
    stderr = float(np.sqrt(max(win_rate * (1 - win_rate), 1e-9) / n))
    return {
        f"eval/{prefix}": win_rate,
        f"eval/{prefix}_stderr": stderr,
        f"eval/{prefix}_score_for": m("score_for"),
        f"eval/{prefix}_score_against": m("score_against"),
        f"eval/{prefix}_return_for": m("return_for"),
        f"eval/{prefix}_return_against": m("return_against"),
        f"eval/{prefix}_caidas_for": m("caidas_for"),
        f"eval/{prefix}_caidas_against": m("caidas_against"),
        f"eval/{prefix}_limpias_for": m("limpias_for"),
        f"eval/{prefix}_limpias_against": m("limpias_against"),
        f"eval/{prefix}_plies": m("plies"),
        f"eval/{prefix}_n_games": float(n),
    }


def member_metrics(metrics: Mapping[str, float], name: str) -> Dict[str, float]:
    """
    The canonical numbers for ONE member, keyed by suffix (`""`, `"_stderr"`, ...).

    Raises KeyError when the member was not evaluated, which is the point: a caller that
    reads a member's numbers is claiming that member was played, and a silent default would
    hide a skipped evaluation.
    """
    return {suffix: metrics[f"eval/{name}{suffix}"] for suffix in METRIC_SUFFIXES}


def format_member_line(metrics: Mapping[str, float], name: str,
                       width: int = 15) -> Optional[str]:
    """One aligned summary line for `name`, or None when the member has no keys."""
    if f"eval/{name}" not in metrics:
        return None
    m = member_metrics(metrics, name)
    return (f"{name:<{width}} {m['']:>6.1%} ±{m['_stderr']:>5.1%}  "
            f"n={m['_n_games']:<3.0f} "
            f"score {m['_score_for']:>5.1f}-{m['_score_against']:<5.1f} "
            f"return {m['_return_for']:>6.1f}-{m['_return_against']:<6.1f} "
            f"caídas {m['_caidas_for']:>4.2f}-{m['_caidas_against']:<4.2f} "
            f"limpias {m['_limpias_for']:>4.2f}-{m['_limpias_against']:<4.2f} "
            f"plies {m['_plies']:>3.0f}")


def league_summary_lines(metrics: Mapping[str, float],
                         names: Optional[Sequence[str]] = None) -> List[str]:
    """Summary lines for every member present in `metrics` (in `LEAGUE_MEMBERS` order)."""
    order = [n for n in (names if names is not None else LEAGUE_MEMBERS)
             if f"eval/{n}" in metrics]
    return [line for line in (format_member_line(metrics, n) for n in order) if line]


# ---------------------------------------------------------------------------
# Parallel games
# ---------------------------------------------------------------------------

def _state_dict(net):
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


def _worker(payload):
    """
    Pool worker: rebuild the net from its weights and play one game against one member.

    The weights travel as a plain state_dict rather than the module, so this works under
    both fork and spawn start methods (macOS defaults to spawn, where the parent's globals
    are not inherited). Each worker pins torch to one thread: ten workers each spawning ten
    threads for 1x64x84 tensor maths is slower than ten single-threaded ones.
    """
    seed, net_team, cfg, state_dict, arch, opponent = payload

    import torch
    torch.set_num_threads(1)

    from network import RebelNet

    net = RebelNet(**arch)
    net.load_state_dict(state_dict)
    net.eval()
    return play_eval_game(net, seed, cfg, net_team, opponent)


def evaluate_league(net, members: Mapping[str, Opponent], games: Mapping[str, int],
                    seed_base: int, cfg: SelfPlayConfig, workers: int = 1) -> Dict[str, float]:
    """
    Evaluate `net` against EVERY member of `members`, returning the merged `eval/*` keys.

    Game `i` of every member gets seed `seed_base + i` and the net plays team `i % 2`: the
    seeds are paired both across successive evaluations and across members (the cheapest
    way to resolve small improvements), and the team assignment alternates. `games` gives
    the game count per member name -- the caller decides that, because the primary members
    and the league extension are sized by different flags.

    `workers > 1` runs all the games of all members in ONE process pool; the default is 1
    (sequential, in this process), which is the simpler and completely deterministic path.
    Both give identical results -- each game is fully determined by its seed and member.

    A member with zero games is not in the output at all (see `match_metrics`).
    """
    state_dict = _state_dict(net)
    arch = net.architecture_config()

    plan: List[Tuple[str, int, int]] = []            # (member, offset into results, n)
    payloads = []
    for name, opponent in members.items():
        n_games = int(games[name])
        plan.append((name, len(payloads), n_games))
        for i in range(n_games):
            payloads.append((seed_base + i, i % 2, cfg, state_dict, arch, opponent))

    results: List[Dict] = []
    if workers and workers > 1 and len(payloads) > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            results = list(pool.imap(_worker, payloads))
    else:
        # In-process: reuse the caller's net instead of rebuilding it per game (and do not
        # touch the process-wide torch thread count, which the trainer shares).
        for seed, net_team, _cfg, _sd, _arch, opponent in payloads:
            results.append(play_eval_game(net, seed, cfg, net_team, opponent))

    metrics: Dict[str, float] = {}
    for name, offset, n_games in plan:
        metrics.update(match_metrics(results[offset:offset + n_games], name))
    return metrics


def evaluate(net, n_games: int, seed_base: int, cfg: SelfPlayConfig,
             workers: int = 1, opponent: Optional[Opponent] = None) -> Dict[str, float]:
    """
    Evaluate `net`'s team against ONE opponent (the heuristic bot by default).

    Kept as the single-member entry point; the league goes through `evaluate_league`.
    """
    opponent = Opponent("heuristic", "heuristic") if opponent is None else opponent
    return evaluate_league(net, {opponent.name: opponent}, {opponent.name: n_games},
                           seed_base, cfg, workers=workers)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(metrics: Dict[str, float], names: Sequence[str], checkpoint: str,
                   elapsed: float) -> None:
    lines = league_summary_lines(metrics, names)
    total = sum(int(metrics[f"eval/{n}_n_games"]) for n in names
                if f"eval/{n}_n_games" in metrics)
    print()
    print("=" * 78)
    print(f"Checkpoint: {checkpoint}")
    print(f"{total} game(s) over {len(lines)} member(s); each member's team alternates "
          f"(team 0 on even games, team 1 on odd ones)")
    print("-" * 78)
    for line in lines:
        print(f"  {line}")
    if not lines:
        print("  no games were played")
    else:
        print(f"  time: {elapsed:.1f}s ({elapsed / total:.1f}s per game)")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a ReBeL checkpoint (search + network) against the league: "
                    "reference / heuristic / initial / prev_checkpoint / random")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="ReBeL checkpoint saved by save_rebel_checkpoint (`best.pt`, ...)")
    parser.add_argument("--members", type=str, default="heuristic",
                        help="comma-separated member names to evaluate, or `all` for every "
                             f"available member. Names: {', '.join(LEAGUE_MEMBERS)} "
                             "(default: heuristic)")
    parser.add_argument("--games", type=int, default=20,
                        help="games for the primary members (reference, heuristic)")
    parser.add_argument("--league-games", type=int, default=30,
                        help="games for initial, prev_checkpoint and random")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR,
                        help="where best.pt / policy_<iteration>.pt live (for reference "
                             "and prev_checkpoint)")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="the run's frozen starting checkpoint (the `initial` member)")
    parser.add_argument("--n-iter", type=int, default=32, help="CFR iterations per decision")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--belief-history", type=int, default=12)
    parser.add_argument("--seed-base", type=int, default=50_000,
                        help="game i uses seed seed_base + i")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=1,
                        help="games in parallel (1 = sequential, in this process)")
    args = parser.parse_args()

    import torch

    from network import load_rebel_checkpoint

    device = torch.device(args.device)
    net = load_rebel_checkpoint(args.checkpoint, device)

    cfg = SelfPlayConfig(n_particles=args.particles, belief_history=args.belief_history,
                         n_iter=args.n_iter, depth=args.depth, seed=args.seed_base,
                         explore=False, device=args.device)

    available, skipped = league_members(save_dir=args.save_dir,
                                        init_checkpoint=args.init_checkpoint,
                                        iteration=None)

    requested = args.members.strip()
    if requested.lower() == "all":
        names = [name for name in LEAGUE_MEMBERS if name in available]
    else:
        names = [part.strip() for part in requested.split(",") if part.strip()]
        unknown = [name for name in names if name not in LEAGUE_MEMBERS]
        if unknown:
            parser.error(f"unknown member(s) {unknown}; choose from "
                         f"{', '.join(LEAGUE_MEMBERS)} or `all`")
        names = [name for name in names if name in available]
        for name in [part.strip() for part in requested.split(",") if part.strip()]:
            if name not in available:
                print(f"[skip] {name}: {skipped.get(name, 'unavailable')}")

    if not names:
        print("nothing to evaluate")
        return

    members = {name: available[name] for name in names}
    games = {name: (args.games if name in PRIMARY_MEMBERS else args.league_games)
             for name in names}

    print(f"Evaluating {args.checkpoint} as a member of the league "
          f"(n_iter={args.n_iter}, depth={args.depth}, particles={args.particles}, "
          f"workers={args.workers})")
    for name in names:
        print(f"  {name:<15} {games[name]} games  ({members[name].kind}"
              f"{': ' + members[name].checkpoint if members[name].checkpoint else ''})")

    start = time.perf_counter()
    metrics = evaluate_league(net, members, games, args.seed_base, cfg, workers=args.workers)
    elapsed = time.perf_counter() - start
    _print_summary(metrics, names, args.checkpoint, elapsed)


if __name__ == "__main__":
    main()
