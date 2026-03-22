"""
card.py — Card and Deck definitions for Cuarenta (40-card Spanish deck).

Spanish deck: 4 suits × 10 ranks, but 8s, 9s, and 10s are removed → 40 cards.
Ranks: 1(As), 2, 3, 4, 5, 6, 7, J(11→Jota), Q(12→Caballo), K(13→Rey)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import List
import random


class Suit(IntEnum):
    BRILLOS = 0      # Coins
    CORAZONES_ROJOS = 1     # Cups
    CORAZONES_NEGROS = 2   # Swords
    TREBOLES = 3    # Clubs


class Rank(IntEnum):
    """
    Pip value = face rank. Used for capture arithmetic.
    J/Q/K are 11/12/13 internally but displayed as Jack/Queen/King.
    """
    AS = 1
    DOS = 2
    TRES = 3
    CUATRO = 4
    CINCO = 5
    SEIS = 6
    SIETE = 7
#  (8, 9 and 10 are removed)
    JACK = 11    # Jack 
    QUEEN = 12 # Queen
    KING = 13     # King


RANK_DISPLAY = {
    Rank.AS: "A",
    Rank.DOS: "2",
    Rank.TRES: "3",
    Rank.CUATRO: "4",
    Rank.CINCO: "5",
    Rank.SEIS: "6",
    Rank.SIETE: "7",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
}

SUIT_DISPLAY = {
    Suit.BRILLOS: "O",
    Suit.CORAZONES_ROJOS: "C",
    Suit.CORAZONES_NEGROS: "E",
    Suit.TREBOLES: "B",
}

ALL_RANKS = [
    Rank.AS, Rank.DOS, Rank.TRES, Rank.CUATRO,
    Rank.CINCO, Rank.SEIS, Rank.SIETE,
    Rank.JACK, Rank.QUEEN, Rank.KING
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