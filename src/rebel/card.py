from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import List
import random


class Suit(IntEnum):
    BRILLOS = 0
    CORAZONES_ROJOS = 1
    CORAZONES_NEGROS = 2
    TREBOLES = 3


class Rank(IntEnum):
    AS = 1
    DOS = 2
    TRES = 3
    CUATRO = 4
    CINCO = 5
    SEIS = 6
    SIETE = 7
#  (8, 9 and 10 are removed)
    JACK = 11
    QUEEN = 12
    KING = 13


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
    Suit.BRILLOS: "B",
    Suit.CORAZONES_ROJOS: "CR",
    Suit.CORAZONES_NEGROS: "CN",
    Suit.TREBOLES: "T",
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
        return int(self.rank)

    @property
    def index(self) -> int:
        """Unique 0–39 index: suit * 10 + rank_position."""
        rank_pos = ALL_RANKS.index(self.rank)
        return int(self.suit) * 10 + rank_pos

    def __str__(self) -> str:
        return f"{RANK_DISPLAY[self.rank]}_{SUIT_DISPLAY[self.suit]}"

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other: Card) -> bool:
        return self.index < other.index


"""
Return all 40 cards in a fresh, unshuffled deck.
"""
def build_deck() -> List[Card]:
    return [Card(rank, suit) for suit in Suit for rank in ALL_RANKS]


"""
Return a shuffled 40-card deck.
"""
def shuffled_deck(rng: random.Random | None = None) -> List[Card]:
    deck = build_deck()
    if rng:
        rng.shuffle(deck)
    else:
        random.shuffle(deck)
    return deck