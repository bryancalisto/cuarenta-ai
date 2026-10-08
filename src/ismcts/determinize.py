# determinize.py
from __future__ import annotations
from typing import List, Dict
import random
import numpy as np

from card import Card, build_deck
from env import CuarentaEnv, CuarentaState, NUM_PLAYERS, HAND_SIZE


class Determinizer:
    """
    Samples a full CuarentaState consistent with the root player's observations.

    Given the root player's info set - their hand, the table, both teams' captures, deck size, per-player turn counts - produce a fully-specified CuarentaState that is consistent with all of that. That means filling in:
    - Each opponent's hand (5 cards minus their plays)
    - Root player's teammate's hand (5 cards minus their plays)
    - The deck (whatever's left)

    Everything else is already known to the root player.

    Usage:
        det = Determinizer(env.state, root_player)
        sampled_state = det.sample(rng)
    """

    def __init__(self, env: CuarentaEnv, root_player: int):
        self.env = env
        self.root = root_player
        self.s = env.state

        all_cards = build_deck()

        # Known to root player: own hand, table, both captures.
        known = (
            set(self.s.hands[root_player])
            | set(self.s.table)
            | set(self.s.cards_captured[0])
            | set(self.s.cards_captured[1])
        )
        # Everything else is unknown: lives in other hands or the deck.
        self.unknown_cards: List[Card] = [c for c in all_cards if c not in known]

        # Sizes we must fill. Note: for the root player's team, their teammate
        # is still an opponent of the root for info-set purposes.
        self.other_hand_sizes: Dict[int, int] = {}
        for p in range(NUM_PLAYERS):
            if p == root_player:
                continue

            expected = HAND_SIZE - self.s.plays_this_deal[p]
            actual = len(self.s.hands[p])
            assert expected == actual, (
                f"plays_this_deal desync for P{p}: "
                f"HAND_SIZE({HAND_SIZE}) - plays({self.s.plays_this_deal[p]}) = {expected}, "
                f"but actual hand size is {actual}"
            )

            # How many cards has p played this deal?
            played = self.s.plays_this_deal[p]
            self.other_hand_sizes[p] = HAND_SIZE - played

        self.deck_size = len(self.s.deck)

        total_needed = sum(self.other_hand_sizes.values()) + self.deck_size
        assert total_needed == len(self.unknown_cards), (
            f"determinizer bookkeeping mismatch: need {total_needed}, "
            f"have {len(self.unknown_cards)}"
        )


    def sample(self, rng: random.Random) -> CuarentaState:
        """Return a plausible CuarentaState from the root player's info set."""
        pool = self.unknown_cards.copy()
        rng.shuffle(pool)

        idx = 0
        hands: Dict[int, List[Card]] = {}

        # Root player's hand is known — copy it.
        hands[self.root] = list(self.s.hands[self.root])

        # Deal out unknown cards to other players' hands.
        for p in range(NUM_PLAYERS):
            if p == self.root:
                continue
            n = self.other_hand_sizes[p]
            hands[p] = pool[idx:idx + n]
            idx += n

        # Rest is the deck.
        deck = pool[idx:idx + self.deck_size]
        assert len(deck) == self.deck_size

        # Reconstruct state. Public fields are copied verbatim; hidden fields are filled from the sample.
        return CuarentaState(
            hands=hands,
            table=list(self.s.table),
            deck=deck,
            scores=list(self.s.scores),
            cards_captured=[list(self.s.cards_captured[0]), list(self.s.cards_captured[1])],
            current_player=self.s.current_player,
            last_placed=self.s.last_placed,
            plays_this_deal=list(self.s.plays_this_deal),
            done=self.s.done,
            winner=self.s.winner,
        )