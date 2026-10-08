"""
The public belief state (PBS): a particle-based belief over the cards that are still
hidden, updated by Bayes' rule as actions are observed.

WHAT A PBS IS HERE
------------------
Cuarenta's only hidden information is which cards sit in which hand -- the table, both
capture piles, the scores, the deck SIZE and the per-seat turn counts are all public. So
a belief state is a probability distribution over deals consistent with everything that
has been observed.

This module represents that distribution as a weighted PARTICLE SET: a list of concrete
deals (full `CuarentaState`s) plus a weight per deal. That is the standard way to carry a
distribution over a space far too large to enumerate (with 40 cards there are vastly more
consistent deals than anyone can tabulate), and it makes ReBeL's public belief update
cheap: reweighting particles by the likelihood of an observed action IS the Bayes update.

TWO THINGS THE BELIEF HAS TO GET RIGHT
--------------------------------------
1. CONDITIONING ON THE ACTING PLAYER. The seat about to act knows its own hand, so its
   belief must condition on it. `refresh()` draws particles whose hand for that seat is
   exactly the real one; the rest of the deal is uniform. That is what makes the root of
   a search a single information set rather than a cloud of them.

2. BAYES ON THE ACTION HISTORY. An opponent playing a card is evidence: they held it, and
   how likely they were to play it says something about the rest of their hand.
   `_posterior_weights()` multiplies each particle's weight by the probability of every
   recently observed action, which is the ReBeL belief update with the current policy as
   the likelihood model. The likelihood for an action comes from, in order:

     * the strategy the actor actually followed, when it is known and covers that hand
       (the solver hands its tables over -- see `record_action`);
     * otherwise the policy network, which generalises to any hand (`policy_fn`);
     * otherwise 1.0, i.e. the action is taken as uninformative.

   A particle's hand at an old decision is reconstructed as "current hand + everything
   that seat has played since", so it always CONTAINS the card that was played -- the
   observed card itself carries no information (everyone saw it). What discriminates
   between particles is the rest of the hand: a hand with which the actor would almost
   never have chosen that card is down-weighted. The size and legality checks in
   `_action_likelihood` are therefore consistency guards (they catch a reconstruction
   that has gone wrong, e.g. across a deal boundary) rather than the main filter.

   With no strategy and no policy available -- say an opponent whose play we cannot model
   -- every likelihood is 1.0 and the belief stays the uniform prior conditioned on the
   public state. That is the honest outcome: nothing was learned from the play.

WHAT THE NETWORK SEES
---------------------
`ownership()`: the posterior marginal probability of each (seat, card) pair, 4 x 40 = 160
floats. This is a lossy summary -- it drops the correlations between hands that the
particles do represent -- and it is the main simplification in this implementation; see
`network.py`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from core import (BELIEF_SEATS, CARD_SLOTS, HAND_SIZE, NUM_PLAYERS, Action, Card,
                  CuarentaState, build_deck, clone_state, env_over,
                  legal_actions_from_state, public_key)
from determinize import Determinizer

# A likelihood of exactly 0 would take the whole belief to zero, so the soft factors are
# floored. The floor is small enough that such a particle is effectively dead anyway.
MIN_LIKELIHOOD = 1e-6

# How many of the most recent actions feed the Bayes update. Keeping all of them would be
# the faithful thing; the last dozen carries nearly all the signal (an old play says
# little about cards that are long gone) and keeps the update cheap.
DEFAULT_HISTORY = 12

# policy_fn(seat, state, legal) -> probabilities aligned with `legal`, or None.
PolicyFn = Callable[[int, CuarentaState, Sequence[Action]], Optional[np.ndarray]]


@dataclass
class ActionRecord:
    """One observed action, with everything needed to score it against a particle."""
    seat: int
    card_index: int
    capture_indices: Tuple[int, ...]
    public_key: tuple
    # The strategy used at the time, keyed by (public_key, seat, hand) with probabilities
    # aligned to that information set's legal-action list. None when the actor was not one
    # of our solvers (e.g. the heuristic bot during evaluation).
    strategy: Optional[Dict] = None


_ALL_CARDS: Optional[List[Card]] = None


def _all_cards() -> List[Card]:
    global _ALL_CARDS
    if _ALL_CARDS is None:
        _ALL_CARDS = build_deck()
    return _ALL_CARDS


def hand_key(cards) -> FrozenSet[int]:
    return frozenset(c.index for c in cards)


def state_from_public(key: tuple, actor_hand: FrozenSet[int]) -> CuarentaState:
    """
    Rebuild enough of a state to enumerate a seat's legal actions from a public key.

    `legal_actions_from_state` reads only the acting seat's hand, the table, the scores,
    `last_placed` and `current_player` -- all of which the public key carries -- so the
    other hands and the deck contents can be left empty. That is what lets the belief
    score an old action against a freshly drawn particle without replaying the game.
    """
    (current_player, table_idx, cap0, cap1, scores, plays, _deck_size, last_idx) = key
    by_index = {c.index: c for c in _all_cards()}
    state = CuarentaState(
        hands={p: [] for p in range(NUM_PLAYERS)},
        table=[by_index[i] for i in table_idx],
        deck=[],                       # contents are irrelevant to legality; size is not
        scores=list(scores),
        cards_captured=[[by_index[i] for i in cap0], [by_index[i] for i in cap1]],
        current_player=current_player,
        last_placed=by_index[last_idx] if last_idx >= 0 else None,
        plays_this_deal=list(plays),
        done=False,
        winner=None,
    )
    state.hands[current_player] = [by_index[i] for i in sorted(actor_hand)]
    return state


def action_index_of(legal: Sequence[Action], card_index: int,
                    capture_indices: Tuple[int, ...]) -> Optional[int]:
    """Position of the (card, capture set) action in a legal-action list, or None."""
    want = sorted(capture_indices)
    for i, action in enumerate(legal):
        if action.card.index != card_index:
            continue
        got = sorted(c.index for c in action.capture) if action.capture else []
        if got == want:
            return i
    return None


class PublicBelief:
    """
    Weighted particle belief over the hidden hands, from the perspective of one seat.

    Lifecycle inside a game (see `selfplay.py`):

        belief = PublicBelief(n_particles=32, rng=rng)
        belief.policy_fn = lambda seat, state, legal: net.policy_probs(state, seat, legal)
        ...
        belief.refresh(true_state, actor)                      # condition on the actor's hand
        solver.solve(true_state, actor, belief)                # search over the particles
        belief.record_action(state_before, actor, action, legal, strategy)   # Bayes step
    """

    def __init__(self, n_particles: int = 32, rng: Optional[random.Random] = None,
                 history: int = DEFAULT_HISTORY):
        self.n_particles = int(n_particles)
        self.rng = rng or random.Random(0)
        self.history = int(history)
        self.particles: List[CuarentaState] = []
        self.weights: np.ndarray = np.zeros(0, dtype=np.float64)
        self.records: List[ActionRecord] = []
        self.observer: Optional[int] = None
        self.policy_fn: Optional[PolicyFn] = None
        self.depleted_updates = 0     # times the update fell back to uniform
        self.resamples = 0

    # ------------------------------------------------------------------ sampling

    def refresh(self, true_state: CuarentaState, observer: int) -> None:
        """
        Re-draw the particles so they are consistent with the current public state AND
        `observer`'s hand, then re-apply the Bayes weights from the action history.

        Re-drawing rather than filtering old particles is what keeps the belief healthy as
        hands shrink: every ply changes the hand sizes and the capture piles, and a fresh
        sample satisfies all of those constraints at once instead of slowly starving a
        fixed particle set.
        """
        self.observer = int(observer)
        self._drop_stale_records(true_state)
        determinizer = Determinizer(env_over(true_state), observer)
        self.particles = [determinizer.sample(self.rng) for _ in range(self.n_particles)]
        self.weights = self._posterior_weights()

    def _drop_stale_records(self, true_state: CuarentaState) -> None:
        """
        Forget records from an earlier deal.

        A new deal re-deals every hand, so evidence gathered before it says nothing about
        the cards held now -- and it actively corrupts the update, because reconstructing
        an old hand from `current hand + cards played since` assumes the whole span sits
        inside one deal (a play from the previous deal is in a capture pile, not in a
        hand, and the turn counters that size the hands have since reset).

        A new deal is visible publicly: every seat's turn count for the deal goes back to
        zero.
        """
        if not self.records:
            return
        fresh_deal = tuple(int(p) for p in true_state.plays_this_deal) == (0, 0, 0, 0)
        stale = any(tuple(int(p) for p in r.public_key[5]) != (0, 0, 0, 0) for r in self.records)
        if fresh_deal and stale:
            self.records.clear()

    # ------------------------------------------------------------------ updates

    def record_action(self, state_before: CuarentaState, seat: int, action: Action,
                      strategy: Optional[Dict] = None) -> None:
        """Record an action that was actually played. The next `refresh` folds it in."""
        capture = tuple(sorted(c.index for c in action.capture)) if action.capture else ()
        self.records.append(ActionRecord(
            seat=int(seat),
            card_index=action.card.index,
            capture_indices=capture,
            public_key=public_key(state_before),
            strategy=strategy,
        ))

    def _hand_at(self, particle: CuarentaState, seat: int, record_pos: int) -> FrozenSet[int]:
        """
        The hand `seat` held when it acted at `self.records[record_pos]`.

        That hand INCLUDES the card played at that position (you can only play a card you
        hold) and every card the seat has played since; cards it played earlier are gone.
        The sizes work out: the current hand has `HAND_SIZE - plays_now` cards, the seat
        has made `plays_now - plays_before` plays from this position onwards, so the
        reconstruction has `HAND_SIZE - plays_before` cards -- exactly the hand size the
        public key says it had.
        """
        cards = set(hand_key(particle.hands[seat]))
        for later in self.records[record_pos:]:
            if later.seat == seat:
                cards.add(later.card_index)
        return frozenset(cards)

    def _action_likelihood(self, record: ActionRecord, particle: CuarentaState,
                           record_pos: int) -> float:
        """
        P(observed action | this particle): the Bayes factor for that particle.

        Combines a HARD constraint (the card must have been in that seat's hand, of the
        right size -- otherwise this particle is impossible) with a SOFT factor (how
        likely the action was under the policy that produced it).
        """
        hand = self._hand_at(particle, record.seat, record_pos)
        expected_size = HAND_SIZE - record.public_key[5][record.seat]
        if len(hand) != expected_size:
            return 0.0

        state = state_from_public(record.public_key, hand)
        legal = legal_actions_from_state(state)
        idx = action_index_of(legal, record.card_index, record.capture_indices)
        if idx is None:
            return 0.0                       # impossible with this hand

        # 1) the strategy the actor was actually following, when it covers this hand
        if record.strategy is not None:
            probs = record.strategy.get((record.public_key, record.seat, hand))
            if probs is not None and idx < len(probs):
                return max(float(probs[idx]), MIN_LIKELIHOOD)

        # 2) the policy network, which generalises to hands the search never visited
        if self.policy_fn is not None:
            probs = self.policy_fn(record.seat, state, legal)
            if probs is not None and idx < len(probs):
                return max(float(probs[idx]), MIN_LIKELIHOOD)

        # 3) uninformative
        return 1.0

    def _posterior_weights(self) -> np.ndarray:
        """Posterior over the current particles: likelihoods of the recent actions."""
        n = len(self.particles)
        weights = np.ones(n, dtype=np.float64)
        recent = self.records[-self.history:] if self.history > 0 else []
        offset = len(self.records) - len(recent)
        for pos, record in enumerate(recent):
            record_pos = offset + pos
            for i, particle in enumerate(self.particles):
                if weights[i] == 0.0:
                    continue
                weights[i] *= self._action_likelihood(record, particle, record_pos)

        total = weights.sum()
        if total <= 0.0:
            # Every particle is inconsistent with what was observed -- in practice this
            # means the hard constraints from cards played early in the window conflict
            # with the fresh sample, or the strategies did not cover any of these hands.
            # Falling back to uniform keeps the search running, and it is counted so the
            # condition is visible in the logs rather than silent.
            self.depleted_updates += 1
            return np.full(n, 1.0 / n, dtype=np.float64)
        return weights / total

    # ------------------------------------------------------------------ maintenance

    def effective_sample_size(self) -> float:
        """1 / sum(w^2): how many particles the belief is really standing on."""
        if self.weights.size == 0 or not np.any(self.weights):
            return 0.0
        return float(1.0 / np.square(self.weights).sum())

    def maybe_resample(self, min_ess_fraction: float = 0.3) -> None:
        """
        Systematic resampling when the weights collapse onto a few particles.

        Without it the belief degenerates: one particle keeps all the mass, the search
        stops seeing the uncertainty it is supposed to plan against, and the belief
        behaves like a single determinization.
        """
        if self.weights.size == 0 or not np.any(self.weights):
            return
        if self.effective_sample_size() >= min_ess_fraction * self.n_particles:
            return
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(self.weights)
        indices = np.clip(np.searchsorted(cumulative, positions), 0, self.n_particles - 1)
        self.particles = [clone_state(self.particles[i]) for i in indices]
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.resamples += 1

    # ------------------------------------------------------------------ readout

    def ownership(self) -> np.ndarray:
        """
        Posterior marginal P(card c is in seat p) as a [4, 40] float32 matrix.

        The weights sum to 1, so accumulating them over the particles that place a card in
        a hand is already the probability; whatever is left over for a card is its chance
        of still being in the deck.
        """
        out = np.zeros((BELIEF_SEATS, CARD_SLOTS), dtype=np.float32)
        if not self.particles:
            return out
        weights = self.weights if self.weights.size == len(self.particles) else None
        for i, particle in enumerate(self.particles):
            w = 1.0 if weights is None else float(weights[i])
            if w == 0.0:
                continue
            for seat in range(BELIEF_SEATS):
                for card in particle.hands[seat]:
                    out[seat, card.index] += w
        return np.clip(out, 0.0, 1.0)

    def sample_particle(self) -> CuarentaState:
        """Draw a deal from the belief; the solver uses this as its chance node."""
        if not self.particles:
            raise RuntimeError("belief has no particles; call refresh() first")
        if self.weights.size != len(self.particles):
            return clone_state(self.rng.choice(self.particles))
        pick = self.rng.choices(range(len(self.particles)),
                                weights=self.weights.tolist(), k=1)[0]
        return clone_state(self.particles[pick])

    def summary(self) -> str:
        return (f"belief: {len(self.particles)} particles, ESS "
                f"{self.effective_sample_size():.1f}, {len(self.records)} actions, "
                f"{self.depleted_updates} depleted, {self.resamples} resamples")
