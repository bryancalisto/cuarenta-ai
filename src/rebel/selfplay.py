"""
Self-play game generation for the ReBeL pipeline.

WHAT A GAME HERE IS
-------------------
Every seat is played by the SAME search (`ReBeLSolver`), which is the ReBeL version of
AlphaZero's self-play: the search is the policy, and its root strategy becomes the
training target for the network. Two differences from `src/ismcts/selfplay.py` are worth
stating, because they are the point of the whole pipeline:

  * the search plans against a BELIEF (a weighted particle set over the hidden hands),
    not against one sampled determinization, so the strategy it returns is the one it
    would play without knowing the opponents' cards;
  * the action actually taken is sampled from that strategy (`explore=True`), so the
    games cover the branches of the tree the network would otherwise never see. With
    `explore=False` the search plays its argmax, which is what evaluation wants.

THE TWO TARGETS PER DECISION
----------------------------
Every ply contributes one training example, and the example carries exactly the two
things the network learns from:

    target_probs   the search's average strategy at the root [n_legal]  -> policy target
    z              +1 if the acting seat's TEAM won the game, else -1   -> value target

`z` is attached AFTER the game, from the acting team's point of view, because utility in
this game is the win: both seats of a team share it (they cooperate), and the per-move
scores (+2 caída, +50 win) are already folded into who wins. Utilities are therefore
zero-sum across teams, which is what makes one value head enough for the whole table.
No discounting: a game is short and the payoff is terminal.

THE INPUT VECTOR IS BUILT ONCE, HERE
------------------------------------
`obs_input` is the 368-float vector the network was fed for that decision: the 208-dim
observation of the acting seat plus the 160 belief marginals (4 seats x 40 cards) at that
moment. It is stored verbatim rather than recomputed at training time, because the belief
moves with every ply: recomputing it later would train the network on a belief it never
saw when it produced the target.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Scripts are run as `python3 src/rebel/<script>.py` from the repository root, so this
# file's own directory must be on sys.path for the bare sibling imports below (the same
# trick src/imitation_learning uses). Anything else would let a same-named module from
# another pipeline (`env.py`, `network.py`, ...) shadow this one.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from belief import DEFAULT_HISTORY, PublicBelief                    # noqa: E402
from core import (BELIEF_DIM, OBS_DIM, TEAMS, Action, CuarentaEnv,  # noqa: E402
                  CuarentaState, action_features_matrix, clone_state, flat_obs_for)
from solver import ReBeLSolver                                      # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SelfPlayConfig:
    """
    Everything `play_game` needs that is not the network itself.

    The defaults are the ones the pipeline is meant to run at: a 3-ply lookahead with 32
    CFR iterations over 16 belief particles. `belief_history` is how many recent actions
    the belief's Bayes update scores (see belief.DEFAULT_HISTORY).
    """
    games: int = 4                      # games per training iteration
    n_particles: int = 16               # belief particles (the search's chance node)
    belief_history: int = DEFAULT_HISTORY
    n_iter: int = 32                    # CFR iterations per decision
    depth: int = 3                      # plies of search before the network values a leaf
    seed: int = 0
    explore: bool = True                # sample from the search's strategy (self-play)
    device: str = "cpu"                 # only affects the solver's network calls


# ---------------------------------------------------------------------------
# One decision
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """
    One training example: what the network saw, and what the search decided.

    Shapes are pinned to the shared layout so a batch is a plain `np.stack`:
    `obs_input [368]`, `feats [64, 84]`, `mask [64]`, `target_probs [n_legal]`.
    """
    acting_seat: int
    acting_team: int
    obs_input: np.ndarray      # float32 [368] = 208 observation + 160 belief marginals
    feats: np.ndarray          # float32 [64, 84] candidate-action features
    mask: np.ndarray           # bool [64], legal slots packed first
    n_legal: int
    target_probs: np.ndarray   # float32 [n_legal] = the search's root strategy
    search_value: float        # the search's root value, from the acting TEAM's view
    z: float = 0.0             # +1 / -1, filled in when the game ends


@dataclass
class GameRecord:
    """A finished self-play game: its decisions, its result and its bookkeeping."""
    decisions: List[Decision]
    winner: Optional[int]          # team that won (0 or 1); None only if unfinished
    scores: Tuple[int, int]        # final scores, [team 0, team 1]
    plies: int
    seed: int = 0
    belief_depleted: int = 0       # refresh() calls that had to fall back to uniform
    belief_resamples: int = 0
    leaf_evals: int = 0            # network calls made by the search, over the game

    @property
    def score_diff(self) -> float:
        """Team 0's final score minus team 1's -- the self-play balance signal."""
        return float(self.scores[0] - self.scores[1])

    @property
    def search_values(self) -> List[float]:
        return [d.search_value for d in self.decisions]

    def summary(self) -> str:
        return (f"game(seed={self.seed}): {self.plies} plies, winner=team {self.winner}, "
                f"scores {self.scores[0]}-{self.scores[1]}, {len(self.decisions)} decisions, "
                f"{self.leaf_evals} leaf evals")


# ---------------------------------------------------------------------------
# The belief's likelihood model
# ---------------------------------------------------------------------------

def belief_policy_fn(net, belief: PublicBelief, allowed_teams: Optional[Sequence[int]] = None):
    """
    The likelihood model the belief uses to score actions the search never visited.

    WHY THIS EXISTS
    ---------------
    `PublicBelief._posterior_weights` multiplies each particle's weight by the
    probability of every recently observed action. When the actor was one of our solvers
    the strategy table covers that hand exactly, so no model is needed. But the belief
    re-draws its particles every ply, and a re-drawn particle holds a hand the search
    never expanded -- for those, the strategy table has no entry and the network is the
    only thing that can say "how likely was this player to play that card, holding these
    cards?". That is the ReBeL belief update: the current policy is the likelihood model.

    `allowed_teams` limits which seats this model is willing to score. During evaluation
    the opponent is the heuristic bot, whose policy is NOT this network; scoring its
    plays with the network's likelihood would push the belief in the wrong direction, so
    those records are left uninformative (returning None makes belief.py fall back to a
    likelihood of 1.0, which is the honest "nothing was learned" case).

    The input vector is built exactly as `play_game` builds the root's, so the model the
    belief uses is the same function of (observation, belief marginals) the search used.
    """
    def policy_fn(seat: int, state: CuarentaState, legal: Sequence[Action]):
        if not legal:
            return None
        if allowed_teams is not None and TEAMS[seat] not in allowed_teams:
            return None
        obs = np.concatenate([flat_obs_for(seat, state), belief.ownership().reshape(-1)])
        feats, _mask = action_features_matrix(list(legal), state)
        return net.policy_probs(obs.astype(np.float32), feats, len(legal))

    return policy_fn


def build_input(seat: int, state: CuarentaState, belief: PublicBelief) -> np.ndarray:
    """
    The 368-float network input for `seat`: observation + belief marginals.

    Single source of truth: `play_game`, the belief's likelihood model and the evaluator
    all come through here, so a training example can never be built from a differently
    arranged vector than the one the search saw.
    """
    obs = np.concatenate([flat_obs_for(seat, state),
                          belief.ownership().reshape(-1)]).astype(np.float32)
    # A layout drift here would not crash anything -- the network would quietly consume a
    # differently arranged vector -- so the width is checked where it is built.
    assert obs.shape == (OBS_DIM + BELIEF_DIM,), obs.shape
    return obs


# ---------------------------------------------------------------------------
# Playing one game
# ---------------------------------------------------------------------------

def choose_action(probs: np.ndarray, rng: random.Random, explore: bool = True) -> int:
    """
    Slot to play: sample from the search's strategy, or take its argmax.

    Sampling is what makes self-play data useful -- an argmax-only agent replays the same
    line every time and the network never learns why the alternatives are worse. This is
    also the one place where stochasticity enters the search root; the search's own
    internals keep their own rng.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if not explore:
        return int(np.argmax(probs))
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return int(rng.randrange(len(probs)))
    return int(rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0])


def play_game(net, seed: int, cfg: SelfPlayConfig,
              rng: Optional[random.Random] = None) -> GameRecord:
    """
    Play one full game in which every seat is the ReBeL search.

    The loop is the whole pipeline in miniature, and the order of the four steps matters:

      1. `belief.refresh(state, seat)` conditions the particles on the ACTING seat's hand,
         so the search root is one information set rather than a cloud of them;
      2. the network input is built once (observation + belief marginals) and kept, so the
         training example is exactly what the search was conditioned on;
      3. `solver.solve` runs the CFR search over deals drawn from that belief and returns
         the root strategy and value;
      4. the action is recorded for the next Bayes update -- together with the search's
         strategy table, which is how the belief knows how likely this play was given a
         hand -- and only then applied to the environment.

    Step 4 must read the state BEFORE `env.step` mutates it (the env's state object is
    updated in place), which is why the clone happens here.
    """
    rng = rng if rng is not None else random.Random(seed)

    env = CuarentaEnv(seed=seed)
    env.reset(seed=seed)

    belief = PublicBelief(n_particles=cfg.n_particles, rng=rng, history=cfg.belief_history)
    belief.policy_fn = belief_policy_fn(net, belief)
    solver = ReBeLSolver(net=net, n_iter=cfg.n_iter, depth=cfg.depth, rng=rng,
                         device=cfg.device)

    decisions: List[Decision] = []
    leaf_evals = 0

    while not env.state.done:
        state = env.state
        seat = int(state.current_player)
        legal = env.legal_actions()

        belief.refresh(state, seat)

        obs_input = build_input(seat, state, belief)
        # `action_features_matrix` already returns float32 [64, 84] with the legal
        # actions packed into the first slots; that layout is what the net consumes.
        feats, mask = action_features_matrix(legal, state)

        result = solver.solve(state, seat, belief)
        leaf_evals += int(result.leaf_evals)

        pick = choose_action(result.root_probs, rng, explore=cfg.explore)
        decisions.append(Decision(
            acting_seat=seat,
            acting_team=int(TEAMS[seat]),
            obs_input=obs_input,
            feats=feats,
            mask=mask,
            n_legal=len(legal),
            target_probs=np.asarray(result.root_probs, dtype=np.float32),
            search_value=float(result.root_value),
        ))

        # The state the belief sees must be the position the action was chosen in, before
        # the env advances: `env.step` mutates `env.state` in place (and a new deal
        # replaces every hand), so the record is taken from a clone right here.
        belief.record_action(clone_state(state), seat, legal[pick], strategy=result.strategy)
        env.step(pick)

    winner = env.state.winner
    for decision in decisions:
        # Utility is the win, from the acting TEAM's point of view: teammates share it,
        # and the two teams' utilities are exact opposites.
        decision.z = 1.0 if winner is not None and winner == decision.acting_team else -1.0

    return GameRecord(
        decisions=decisions,
        winner=winner,
        scores=(int(env.state.scores[0]), int(env.state.scores[1])),
        plies=len(decisions),
        seed=int(seed),
        belief_depleted=int(belief.depleted_updates),
        belief_resamples=int(belief.resamples),
        leaf_evals=leaf_evals,
    )


def play_games(net, cfg: SelfPlayConfig, base_seed: Optional[int] = None,
               progress=None) -> List[GameRecord]:
    """
    Generate `cfg.games` games, seeded `base_seed + i` (or `cfg.seed + i`).

    Seeds are derived rather than shared so a run is reproducible and so two iterations
    never replay the same deals.
    """
    base = cfg.seed if base_seed is None else int(base_seed)
    records: List[GameRecord] = []
    for i in range(cfg.games):
        record = play_game(net, base + i, cfg)
        records.append(record)
        if progress is not None:
            progress(record, i)
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def selfplay_metrics(records: Sequence[GameRecord]) -> Dict[str, float]:
    """
    Aggregate self-play games into the `selfplay/*` family the trainer logs.

    Names match `src/ppo/eval_harness.py` and `src/ismcts/train.py` so the three
    pipelines' curves are directly comparable. `win_rate_team0` is the balance check: it
    should hover near 0.5 in self-play, since both teams are the same agent.
    """
    if not records:
        return {}
    n = len(records)
    decided = [r for r in records if r.winner is not None]
    values = [d.search_value for r in records for d in r.decisions]
    n_decisions = sum(len(r.decisions) for r in records) or 1
    return {
        "selfplay/win_rate_team0": (sum(1 for r in decided if r.winner == 0)
                                    / max(len(decided), 1)),
        "selfplay/plies": sum(r.plies for r in records) / n,
        "selfplay/score_diff": sum(r.score_diff for r in records) / n,
        # Per-game belief health: how often the Bayes update had to give up and fall back
        # to uniform. A non-zero value here means the belief is being corrupted, not that
        # the agent is playing badly -- it should stay at 0.
        "selfplay/belief_depleted": sum(r.belief_depleted for r in records) / n,
        # Leaf evaluations per decision, i.e. the search's cost per move.
        "selfplay/leaf_evals": sum(r.leaf_evals for r in records) / n_decisions,
        "selfplay/search_value_mean": (sum(values) / len(values)) if values else 0.0,
        # Mean |value|: how much the search thinks it knows. Near 0 means every position
        # looks like a coin flip to it (which is what an untrained value head produces).
        "selfplay/search_value_abs_mean": ((sum(abs(v) for v in values) / len(values))
                                           if values else 0.0),
    }


# ---------------------------------------------------------------------------
# Manual run: time a single game (this doubles as the performance report)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Play self-play games with ReBeL")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--n-iter", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--belief-history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--argmax", action="store_true",
                        help="play the search's argmax instead of sampling (no exploration)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import torch

    from network import RebelNet, load_rebel_checkpoint

    torch.set_num_threads(1)
    device = torch.device(args.device)
    if args.checkpoint:
        net = load_rebel_checkpoint(args.checkpoint, device)
    else:
        net = RebelNet().to(device).eval()

    cfg = SelfPlayConfig(games=args.games, n_particles=args.particles,
                         belief_history=args.belief_history, n_iter=args.n_iter,
                         depth=args.depth, seed=args.seed, explore=not args.argmax,
                         device=args.device)

    start = time.perf_counter()
    records = play_games(net, cfg, progress=lambda r, i: print(f"  {r.summary()}"))
    elapsed = time.perf_counter() - start

    print(f"\n{args.games} game(s) in {elapsed:.1f}s "
          f"({elapsed / max(args.games, 1):.1f}s per game, "
          f"n_iter={args.n_iter}, depth={args.depth}, particles={args.particles})")
    for key, value in sorted(selfplay_metrics(records).items()):
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
