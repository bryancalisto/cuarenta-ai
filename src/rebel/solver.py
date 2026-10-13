"""
The ReBeL search: a depth-limited subgame solved by CFR, with the network valuing the
leaves. This is the piece that makes the pipeline ReBeL rather than determinized MCTS.

WHY CFR INSTEAD OF PUCT
-----------------------
`src/ismcts` runs PUCT on sampled determinizations and averages the visit counts of the
root. That treats each determinization as a perfect-information game, which is known to be
unsound with hidden information: the search can decide that a bluff is pointless because
in the sampled world the opponent always had the card to punish it, and averaging
argmaxes produces a strategy neither player would play. CFR instead solves the subgame as
a game: it computes regrets against the opponent's whole range and returns a MIXED
strategy, so a play that is only sometimes right can still be correct on average.

The structure is CFR-D: the tree is cut at a fixed depth and the leaves are valued by the
network, exactly like AlphaZero's value leaves -- the difference is that the root
strategy comes from regret matching rather than from visit counts.

HOW THE BELIEF ENTERS
---------------------
Two ways, both from `belief.py`:

  * the chance node: every traversal starts from a deal drawn from the belief, so the
    search plans against the whole range of possible opponents' hands rather than one
    sampled world;
  * the leaf values: the network is fed the belief marginals along with the acting seat's
    observation, so a leaf is valued as "this position, given what we believe about the
    hidden cards".

The belief at the ROOT fixes the acting seat's own hand (see `PublicBelief.refresh`), so
the root information set is a single information set and its average strategy is the
answer the search returns.

WHAT THE VALUE IS
-----------------
Utility is the GAME OUTCOME: +1 if the acting team wins the game, -1 if it loses. The
env's dense per-move rewards (+1 per captured card, +2 caida, +2 limpia, WIN_BONUS=20 for
the win) are NOT added into the backup -- they are scoring events that influence who
eventually wins, which the search accounts for through the dynamics, not utility in their
own right. Keeping the value on the win/loss scale is what makes it comparable across
depths and what `tanh` in the network can represent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import (TEAMS, CuarentaState, action_features_matrix, flat_obs_for,
                  legal_actions_from_state, public_key, step_state)
from belief import PublicBelief

# A node key identifies an information set: the public state, the seat to act, and that
# seat's hand. Two different hands are two different decisions even at the same table.
NodeKey = Tuple[tuple, int, frozenset]


@dataclass
class SolveResult:
    """What the search hands back for one decision."""
    root_probs: np.ndarray          # aligned with the root's legal-action list
    root_value: float               # from the ROOT SEAT's team, in [-1, 1]
    iterations: int
    leaf_evals: int
    depth: int
    explored_nodes: int
    # Average strategy per information set, keyed exactly as the belief expects:
    # (public_key, seat, frozenset(hand)) -> probabilities aligned with that information
    # set's legal-action list. Handed to `PublicBelief.record_action` so the next belief
    # update can ask "how likely was this play, given this hand?" -- the ReBeL belief
    # update uses the policy that was just computed.
    strategy: Dict[NodeKey, np.ndarray] = field(default_factory=dict)


class ReBeLSolver:
    """
    Depth-limited CFR over deals sampled from a public belief state.

    One instance can be reused across decisions (the tables are rebuilt per `solve`); the
    rng is the only state that carries over, which keeps successive searches independent
    but reproducible from a seed.
    """

    def __init__(self, net=None, n_iter: int = 32, depth: int = 3,
                 rng: Optional[random.Random] = None, device: str = "cpu",
                 strategy_averaging: str = "uniform"):
        """
        `strategy_averaging` decides which strategy the search returns:

          * "uniform" (plain CFR, the default) -- every iteration counts the same.
          * "linear" (CFR+) -- iteration `t` is weighted by `t`, so the later, better
            informed strategies dominate the average.
          * "current" -- the last iteration's regret-matched strategy, no averaging.

        "linear" exists to test the guess that the early, still-uniform iterations were
        diluting the returned strategy. MEASURED: it makes no difference to playing
        strength (search + bootstrapped net vs the heuristic: 21.9% linear vs 25.0%
        uniform, 32 games each, +-7.5% se), so the default stays plain and the knob is
        left for anyone revisiting it at higher iteration counts. Whatever is wrong with
        the search, it is not the averaging weight.
        """
        assert strategy_averaging in ("uniform", "linear", "current"), strategy_averaging
        self.net = net
        self.n_iter = int(n_iter)
        self.depth = int(depth)
        self.rng = rng or random.Random(0)
        self.device = device
        self.strategy_averaging = strategy_averaging
        self.regrets: Dict[NodeKey, np.ndarray] = {}
        self.strategy_sum: Dict[NodeKey, np.ndarray] = {}
        self._belief_features: Optional[np.ndarray] = None
        self.leaf_evals = 0
        self.nodes_visited = 0
        self._iteration = 0

    # ------------------------------------------------------------------ entry point

    def solve(self, true_state: CuarentaState, root_seat: int,
              belief: PublicBelief) -> SolveResult:
        """
        Solve the subgame at `true_state` for the seat `root_seat`.

        `belief` must already be refreshed for this decision (`refresh(true_state,
        root_seat)`), so that every particle agrees with the root seat's hand.
        """
        import torch

        root_team = TEAMS[root_seat]
        root_legal = legal_actions_from_state(true_state)
        root_key: NodeKey = (public_key(true_state), int(root_seat),
                             frozenset(c.index for c in true_state.hands[root_seat]))

        self.regrets = {}
        self.strategy_sum = {}
        self.leaf_evals = 0
        self.nodes_visited = 0
        self._belief_features = belief.ownership().reshape(-1)

        root_values: List[float] = []
        for self._iteration in range(1, self.n_iter + 1):
            # Alternate whose regrets are updated, one traversal each -- standard for a
            # two-player zero-sum game (here the two "players" are the two TEAMS, since
            # teammates share a payoff and cooperate).
            for team in (0, 1):
                state = belief.sample_particle()
                with torch.no_grad():
                    value = self._walk(state, team, depth=0, reach=1.0)
                if team == root_team:
                    root_values.append(value)

        probs = self._average_strategy(root_key, len(root_legal))
        return SolveResult(
            root_probs=probs,
            root_value=float(np.mean(root_values)) if root_values else 0.0,
            iterations=self.n_iter,
            leaf_evals=self.leaf_evals,
            depth=self.depth,
            explored_nodes=self.nodes_visited,
            strategy=self.average_strategy_table(),
        )

    def average_strategy_table(self) -> Dict[NodeKey, np.ndarray]:
        """
        Every information set's average strategy, normalised, ready for the belief.

        Only keys that were actually accumulated are included; the belief treats a missing
        key as "this information set was never searched" and falls back to the network.
        """
        table: Dict[NodeKey, np.ndarray] = {}
        for key, total in self.strategy_sum.items():
            s = float(total.sum())
            if s > 0.0:
                table[key] = total / s
        return table

    # ------------------------------------------------------------------ the walk

    def _walk(self, state: CuarentaState, team: int, depth: int, reach: float) -> float:
        """One CFR traversal; returns the value to `team` from this node."""
        self.nodes_visited += 1

        if state.done:
            return 1.0 if state.winner == team else -1.0
        if depth >= self.depth:
            return self._leaf_value(state, team)

        seat = state.current_player
        legal = legal_actions_from_state(state)
        key: NodeKey = (public_key(state), int(seat),
                        frozenset(c.index for c in state.hands[seat]))
        strategy = self._regret_matching(key, len(legal))

        if TEAMS[seat] == team:
            # The traverser: try every action, and charge regret against the value it got.
            values = np.empty(len(legal), dtype=np.float64)
            for i, action in enumerate(legal):
                child, _reward, _done, _actor = step_state(state, action, self.rng)
                values[i] = self._walk(child, team, depth + 1, reach * float(strategy[i]))
            node_value = float(np.dot(strategy, values))
            self.regrets[key] += values - node_value
            if self.strategy_averaging != "current":
                weight = 1.0 if self.strategy_averaging == "uniform" else float(self._iteration)
                self.strategy_sum[key] += weight * reach * strategy
            return node_value

        # The opponent: sample one action from their current strategy (external sampling),
        # so the cost stays linear in the branching factor instead of quadratic.
        pick = self.rng.choices(range(len(legal)), weights=strategy.tolist(), k=1)[0]
        child, _reward, _done, _actor = step_state(state, legal[pick], self.rng)
        return self._walk(child, team, depth + 1, reach)

    # ------------------------------------------------------------------ values

    def _leaf_value(self, state: CuarentaState, team: int) -> float:
        """
        Value of a cut-off position, from `team`'s point of view.

        The network predicts the value for the seat to act, so it is negated when that
        seat belongs to the other team. With no network loaded the leaf is worth 0 --
        i.e. "unknown", which is the honest answer and still lets the search use the
        part of the tree it can actually see to the end.
        """
        if self.net is None:
            return 0.0
        self.leaf_evals += 1
        seat = state.current_player
        obs = np.concatenate([flat_obs_for(seat, state), self._belief_features])
        legal = legal_actions_from_state(state)
        feats, _mask = action_features_matrix(legal, state)
        value = self.net.value_of(obs, feats)
        return value if TEAMS[seat] == team else -value

    # ------------------------------------------------------------------ tables

    def _regret_matching(self, key: NodeKey, n_actions: int) -> np.ndarray:
        """
        Current strategy: regrets clipped at zero, normalised; uniform when nothing is
        positive yet (the standard CFR starting point).
        """
        regrets = self.regrets.get(key)
        if regrets is None:
            regrets = np.zeros(n_actions, dtype=np.float64)
            self.regrets[key] = regrets
            self.strategy_sum[key] = np.zeros(n_actions, dtype=np.float64)
        elif regrets.shape[0] != n_actions:
            # Same public state and seat but a different action count means the hand was
            # reconstructed differently -- a bug worth failing on rather than papering over.
            raise RuntimeError(
                f"information set {key[1]} has {regrets.shape[0]} actions but {n_actions} "
                "were generated; the hand key and the legal-action list disagree"
            )
        positive = np.maximum(regrets, 0.0)
        total = positive.sum()
        if total > 0.0:
            return positive / total
        return np.full(n_actions, 1.0 / n_actions, dtype=np.float64)

    def _average_strategy(self, key: NodeKey, n_actions: int) -> np.ndarray:
        """
        The strategy the search returns.

        "current" asks regret matching for the final iterate instead of averaging; the other
        two modes read the (weighted) average accumulated during the traversals.
        """
        if self.strategy_averaging == "current":
            return self._regret_matching(key, n_actions)
        total = self.strategy_sum.get(key)
        if total is None or total.sum() <= 0.0:
            return np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
        return total / total.sum()


# ---------------------------------------------------------------------------
# A toy game, used by the tests to check that the CFR core converges
# ---------------------------------------------------------------------------

# Kuhn poker, in the form the solver cares about: 3 cards (J=0 < Q=1 < K=2), one each,
# ante 1, each player acts once, and a bet can be called or folded. Its equilibrium is
# known exactly, which is why it is worth carrying a toy game here: it is the only way to
# tell "the CFR code is right" from "Cuarenta seems to play okay", and a broken
# regret/strategy update would otherwise just look like a weak network.
#
# Histories are sequences of tokens, and which tokens are available depends on where you
# are -- "c" checks and "b" bets, but the passive reply to a bet is "p" (fold) and the
# aggressive one is "b" (call). Encoding both as "p" (as an earlier draft did) makes some
# lines never terminate.
KUHN_TERMINALS = {"cc", "bp", "bb", "cbp", "cbb"}


def kuhn_tokens(history: str) -> Tuple[str, str]:
    """(passive, aggressive) tokens available at `history`."""
    return ("c", "b") if history in ("", "c") else ("p", "b")


def _kuhn_value(history: str, cards: Sequence[int]) -> float:
    """Payoff to player 0 of a finished betting sequence."""
    showdown = 1.0 if cards[0] > cards[1] else -1.0
    if history == "cc":
        return showdown                    # both checked: pot 2, winner nets 1
    if history == "bp":
        return 1.0                         # player 1 folded to the bet
    if history == "bb":
        return 2.0 * showdown              # called: pot 4, winner nets 2
    if history == "cbp":
        return -1.0                        # player 0 folded to player 1's bet
    if history == "cbb":
        return 2.0 * showdown              # player 0 called
    raise ValueError(f"not a terminal history: {history!r}")


def kuhn_poker_solve(n_iter: int = 20000, seed: int = 0) -> Dict[str, float]:
    """
    Run the SAME algorithm the Cuarenta solver runs -- depth-unlimited external-sampling
    MCCFR with regret matching -- on Kuhn poker, and report quantities whose true values
    are known.

    Expected results: the game value to player 0 is -1/18 = -0.0556, and player 0 bluffs
    with a jack one time in three (1/3 = 0.333) at the opening decision.

    The average strategy is evaluated exactly (over the six possible deals) rather than by
    sampling, so the value it reports is not itself noisy.
    """
    rng = random.Random(seed)
    regrets: Dict[tuple, np.ndarray] = {}
    strategy_sum: Dict[tuple, np.ndarray] = {}

    def strategy_at(key) -> np.ndarray:
        r = regrets.setdefault(key, np.zeros(2))
        strategy_sum.setdefault(key, np.zeros(2))
        positive = np.maximum(r, 0.0)
        total = positive.sum()
        return (positive / total) if total > 0 else np.full(2, 0.5)

    def walk(history: str, cards: Sequence[int], traverser: int, reach: float) -> float:
        if history in KUHN_TERMINALS:
            value = _kuhn_value(history, cards)
            return value if traverser == 0 else -value
        player = len(history) % 2
        key = (history, cards[player])
        strategy = strategy_at(key)
        passive, aggressive = kuhn_tokens(history)
        if player == traverser:
            values = np.empty(2)
            for a, token in ((0, passive), (1, aggressive)):
                values[a] = walk(history + token, cards, traverser, reach * float(strategy[a]))
            node_value = float(np.dot(strategy, values))
            regrets[key] += values - node_value
            strategy_sum[key] += reach * strategy
            return node_value
        pick = rng.choices((0, 1), weights=strategy.tolist(), k=1)[0]
        return walk(history + (passive if pick == 0 else aggressive), cards, traverser, reach)

    for _ in range(n_iter):
        for traverser in (0, 1):
            deck = [0, 1, 2]
            rng.shuffle(deck)
            walk("", deck, traverser, 1.0)

    def average(key) -> np.ndarray:
        total = strategy_sum.get(key)
        if total is None or total.sum() <= 0:
            return np.full(2, 0.5)
        return total / total.sum()

    def exact_value(history: str, cards: Sequence[int]) -> float:
        """Expected payoff to player 0 when both follow the average strategy."""
        if history in KUHN_TERMINALS:
            return _kuhn_value(history, cards)
        player = len(history) % 2
        probs = average((history, cards[player]))
        passive, aggressive = kuhn_tokens(history)
        value = 0.0
        for a, token in ((0, passive), (1, aggressive)):
            if probs[a] > 0:
                value += float(probs[a]) * exact_value(history + token, cards)
        return value

    import itertools
    deals = [d for d in itertools.permutations((0, 1, 2), 2)]
    game_value = sum(exact_value("", d) for d in deals) / len(deals)
    with_jack = average(("", 0))
    with_king = average(("", 2))
    return {
        "value_to_player0": game_value,
        "bluff_frequency_with_jack": float(with_jack[1]),
        "bet_frequency_with_king": float(with_king[1]),
    }
