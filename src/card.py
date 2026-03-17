"""
card.py — Card and Deck definitions for Cuarenta (40-card Spanish deck).

Spanish deck: 4 suits × 10 ranks, but 8s, 9s, and 10s are removed → 40 cards.
Ranks: 1(As), 2, 3, 4, 5, 6, 7, J(10→Sota), Q(11→Caballo), K(12→Rey)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import List
import random


class Suit(IntEnum):
    OROS = 0      # Coins
    COPAS = 1     # Cups
    ESPADAS = 2   # Swords
    BASTOS = 3    # Clubs


class Rank(IntEnum):
    """
    Pip value = face rank. Used for capture arithmetic.
    J/Q/K are 10/11/12 internally but displayed as Sota/Caballo/Rey.
    """
    AS = 1
    DOS = 2
    TRES = 3
    CUATRO = 4
    CINCO = 5
    SEIS = 6
    SIETE = 7
    SOTA = 10    # Jack  (8 and 9 are removed)
    CABALLO = 11 # Queen
    REY = 12     # King


RANK_DISPLAY = {
    Rank.AS: "A",
    Rank.DOS: "2",
    Rank.TRES: "3",
    Rank.CUATRO: "4",
    Rank.CINCO: "5",
    Rank.SEIS: "6",
    Rank.SIETE: "7",
    Rank.SOTA: "J",
    Rank.CABALLO: "Q",
    Rank.REY: "K",
}

SUIT_DISPLAY = {
    Suit.OROS: "O",
    Suit.COPAS: "C",
    Suit.ESPADAS: "E",
    Suit.BASTOS: "B",
}

ALL_RANKS = [
    Rank.AS, Rank.DOS, Rank.TRES, Rank.CUATRO,
    Rank.CINCO, Rank.SEIS, Rank.SIETE,
    Rank.SOTA, Rank.CABALLO, Rank.REY
]  # 10 ranks × 4 suits = 40 cards


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def value(self) -> int:
        """Pip value used for capture arithmetic."""
        return int(self.rank)

    @property
    def index(self) -> int:
        """Unique 0–39 index: suit * 10 + rank_position."""
        rank_pos = ALL_RANKS.index(self.rank)
        return int(self.suit) * 10 + rank_pos

    def __str__(self) -> str:
        return f"{RANK_DISPLAY[self.rank]}{SUIT_DISPLAY[self.suit]}"

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other: Card) -> bool:
        return self.index < other.index


def build_deck() -> List[Card]:
    """Return all 40 cards in a fresh, unshuffled deck."""
    return [Card(rank, suit) for suit in Suit for rank in ALL_RANKS]


def shuffled_deck(rng: random.Random | None = None) -> List[Card]:
    """Return a shuffled 40-card deck."""
    deck = build_deck()
    if rng:
        rng.shuffle(deck)
    else:
        random.shuffle(deck)
    return deck